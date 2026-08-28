"""Small CPU-only arithmetic tests; no training, checkpoint, backbone or export."""
import torch
from support import OfflineCase
from utils.fc_model import SegmentRankingModel, VideoSegmenterLoss


class ModelMath(OfflineCase):
    """MAN-T02: validates arithmetic, NOT learned anomaly performance."""
    def test_eval_is_repeatable_and_scores_are_bounded(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = SegmentRankingModel(input_dim=8).eval()
            features = torch.randn(3,8)
            with torch.no_grad():
                first, second = model(features), model(features)
            self.assertEqual(tuple(first.shape),(3,1))
            self.assertTrue(torch.equal(first,second))
            self.assertTrue(torch.isfinite(first).all())
            self.assertTrue(((first>=0)&(first<=1)).all())

    def test_loss_terms_are_finite_nonnegative_and_differentiable(self):
        abnormal = torch.tensor([[.2],[.8],[.5]],requires_grad=True)
        normal = torch.tensor([[.1],[.2],[.1]],requires_grad=True)
        terms = VideoSegmenterLoss(.1,.1,.1)(abnormal,normal)
        self.assertEqual(len(terms),4)
        self.assertTrue(all(torch.isfinite(t) and t>=0 for t in terms))
        sum(terms).backward()
        self.assertTrue(torch.isfinite(abnormal.grad).all())
        self.assertTrue(torch.isfinite(normal.grad).all())
