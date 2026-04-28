import json
import pickle
import numpy as np
from collections import defaultdict

from .recommender import Recommender


class ContextualRecommender(Recommender):
    """
    Embedding-based contextual re-ranker (optimized for low latency).

    Key optimizations vs naive version:
    - Artist lookup table built at init (no redis calls per-request)
    - PCA to 64 dims to shrink matmul from ~4096-d to 64-d
    - Session context computed inline with minimal allocations
    - I2I lookups limited to top-2 anchors
    """

    def __init__(
        self,
        listen_history_redis,
        hstu_redis,
        i2i_redis,
        catalog,
        embeddings_path,
        fallback_recommender,
        artist_penalty=0.5,
        pca_dim=64,
    ):
        self.listen_history_redis = listen_history_redis
        self.hstu_redis = hstu_redis
        self.i2i_redis = i2i_redis
        self.catalog = catalog
        self.fallback_recommender = fallback_recommender
        self.artist_penalty = artist_penalty

        self.track_artist = {}
        artist_to_id = {}
        for t in catalog.tracks:
            if t.artist not in artist_to_id:
                artist_to_id[t.artist] = len(artist_to_id)
            self.track_artist[t.track] = artist_to_id[t.artist]

        raw = np.load(embeddings_path)
        if raw.dtype != np.float32:
            raw = raw.astype(np.float32)

        mean = raw.mean(axis=0)
        centered = raw - mean
        rng = np.random.RandomState(42)
        inter_dim = min(pca_dim * 4, centered.shape[1], centered.shape[0])
        omega = rng.randn(centered.shape[1], inter_dim).astype(np.float32)
        y = centered @ omega
        q, _ = np.linalg.qr(y)
        b = q.T @ centered
        _, _, vt = np.linalg.svd(b, full_matrices=False)
        proj = vt[:pca_dim]

        reduced = centered @ proj.T
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = (reduced / norms).astype(np.float32)
        self.emb_dim = pca_dim

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks = set()
        session_artists = defaultdict(int)

        for track, _ in history:
            seen_tracks.add(track)
            aid = self.track_artist.get(track)
            if aid is not None:
                session_artists[aid] += 1

        candidates = set()
        hstu_data = self.hstu_redis.get(user)
        if hstu_data is not None:
            hstu_tracks = self.catalog.from_bytes(hstu_data)
            if isinstance(hstu_tracks, list):
                candidates.update(hstu_tracks)

        if history:
            best_anchors = sorted(history, key=lambda x: x[1], reverse=True)[:2]
            for anchor, _ in best_anchors:
                i2i_data = self.i2i_redis.get(anchor)
                if i2i_data is not None:
                    i2i_recs = pickle.loads(i2i_data)
                    candidates.update(int(t) for t in i2i_recs)

        candidates -= seen_tracks
        max_id = self.embeddings.shape[0] - 1
        candidates = [c for c in candidates if 0 <= c <= max_id]

        if not candidates:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        context = np.zeros(self.emb_dim, dtype=np.float32)
        total_w = 0.0
        for track, listen_time in history:
            if 0 <= track <= max_id and listen_time > 0.05:
                context += listen_time * self.embeddings[track]
                total_w += listen_time

        if total_w == 0:
            if 0 <= prev_track <= max_id:
                context = self.embeddings[prev_track].copy()
            else:
                return candidates[0]
        else:
            context /= total_w

        norm = np.linalg.norm(context)
        if norm > 0:
            context /= norm

        idx = np.array(candidates, dtype=np.int64)
        scores = self.embeddings[idx] @ context

        if session_artists:
            for i, track_id in enumerate(candidates):
                aid = self.track_artist.get(track_id)
                if aid is not None and aid in session_artists:
                    scores[i] *= self.artist_penalty ** session_artists[aid]

        return int(candidates[np.argmax(scores)])

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
