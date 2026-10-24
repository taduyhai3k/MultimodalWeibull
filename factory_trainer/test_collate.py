"""Run in your PyTorch env. Does not read/modify DatasetFactory."""
import unittest
try:
    import torch
    from .trainer import collate_original
except ImportError:
    torch=None

@unittest.skipIf(torch is None,'Requires PyTorch and trainer dependencies')
class CollateTests(unittest.TestCase):
    def test_pathways(self):
        a=(torch.ones(2,768),[torch.tensor([1.,2.])],torch.tensor(0),torch.tensor([3.]),torch.tensor([0.]),('N/A',)*3,torch.tensor([0.,1.]))
        b=(torch.ones(3,768),[torch.tensor([3.,4.])],torch.tensor(1),torch.tensor([5.]),torch.tensor([1.]),('N/A',)*3,torch.zeros(3))
        out=collate_original([(a,0),(b,1)])
        self.assertEqual(tuple(out['x'].shape),(2,3,768))
        self.assertEqual(out['mask'][0].tolist(),[False,True,True])
        torch.testing.assert_close(out['groups'][0],torch.tensor([[1.,2.],[3.,4.]]))
        self.assertEqual(out['time'].tolist(),[3.,5.])
    def test_groups(self):
        a=(torch.ones(2,768),torch.tensor([1.,2.]),torch.tensor([3.]),torch.tensor(0),torch.tensor([3.]),torch.tensor([0.]),('N/A',)*3,torch.zeros(2))
        out=collate_original([(a,0)])
        self.assertEqual(len(out['groups']),2)

if __name__=='__main__':unittest.main()
