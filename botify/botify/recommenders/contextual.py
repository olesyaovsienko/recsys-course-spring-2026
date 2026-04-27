import json
import pickle
import numpy as np
from collections import defaultdict

from .recommender import Recommender


class ContextualRecommender(Recommender):
    """
    Embedding-based contextual re-ranker.

    Combines candidates from multiple sources (HSTU user recs, I2I neighbors)
    and ranks them using cosine similarity between a session context vector
    and candidate track embeddings.

    The session context is computed as a weighted average of embeddings
    of recently listened tracks, where weights are the listen times.

    Artist diversity is enforced by penalizing tracks from artists
    that have already appeared in the session.
    """

    def __init__(
        self,
        listen_history_redis,
        hstu_redis,
        i2i_redis,
        track_redis,
        catalog,
        embeddings_path,
        fallback_recommender,
        artist_penalty=0.5,
        top_k=10,
    ):
        self.listen_history_redis = listen_history_redis
        self.hstu_redis = hstu_redis
        self.i2i_redis = i2i_redis
        self.track_redis = track_redis
        self.catalog = catalog
        self.fallback_recommender = fallback_recommender
        self.artist_penalty = artist_penalty
        self.top_k = top_k

        self.embeddings = np.load(embeddings_path).astype(np.float32)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.normed_embeddings = self.embeddings / norms

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)

        candidates = set()
        seen_tracks = set(track for track, _ in history)

        hstu_data = self.hstu_redis.get(user)
        if hstu_data is not None:
            hstu_tracks = self.catalog.from_bytes(hstu_data)
            if isinstance(hstu_tracks, list):
                candidates.update(hstu_tracks)

        if history:
            track_time = defaultdict(float)
            for track, listened_time in history:
                track_time[track] += listened_time

            sorted_anchors = sorted(track_time.items(), key=lambda x: x[1], reverse=True)
            for anchor, _ in sorted_anchors[:5]:
                i2i_data = self.i2i_redis.get(anchor)
                if i2i_data is not None:
                    i2i_recs = pickle.loads(i2i_data)
                    candidates.update(int(t) for t in i2i_recs)

        candidates -= seen_tracks

        max_track_id = self.embeddings.shape[0] - 1
        candidates = [c for c in candidates if 0 <= c <= max_track_id]

        if not candidates:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        context = self._compute_session_context(history)
        if context is None:
            return candidates[0]

        candidate_indices = np.array(candidates, dtype=np.int64)
        candidate_embeddings = self.normed_embeddings[candidate_indices]

        scores = candidate_embeddings @ context

        artist_counts = self._get_session_artist_counts(history)
        if artist_counts:
            for i, track_id in enumerate(candidates):
                artist = self._get_track_artist(track_id)
                if artist and artist in artist_counts:
                    count = artist_counts[artist]
                    scores[i] *= (self.artist_penalty ** count)

        best_idx = np.argmax(scores)
        return int(candidates[best_idx])

    def _compute_session_context(self, history):
        """Compute weighted average of track embeddings from session history."""
        if not history:
            return None

        max_track_id = self.embeddings.shape[0] - 1
        weighted_sum = np.zeros(self.embeddings.shape[1], dtype=np.float32)
        total_weight = 0.0

        for track, listen_time in history:
            if 0 <= track <= max_track_id and listen_time > 0.05:
                weight = listen_time
                weighted_sum += weight * self.embeddings[track]
                total_weight += weight

        if total_weight == 0:
            for track, _ in history:
                if 0 <= track <= max_track_id:
                    vec = self.embeddings[track]
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        return (vec / norm).astype(np.float32)
            return None

        context = weighted_sum / total_weight
        norm = np.linalg.norm(context)
        if norm == 0:
            return None
        return (context / norm).astype(np.float32)

    def _load_user_history(self, user: int):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)

        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history

    def _get_session_artist_counts(self, history):
        """Count how many times each artist appeared in session."""
        artist_counts = defaultdict(int)
        for track, _ in history:
            artist = self._get_track_artist(track)
            if artist:
                artist_counts[artist] += 1
        return artist_counts

    def _get_track_artist(self, track_id):
        """Get artist name for a track from redis."""
        data = self.track_redis.get(track_id)
        if data is not None:
            track_obj = self.catalog.from_bytes(data)
            return track_obj.artist
        return None
