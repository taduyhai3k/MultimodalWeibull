import tempfile,unittest,zipfile
from pathlib import Path
from prepare_data import extract_archive

class ExtractTests(unittest.TestCase):
    def test_wsi_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);archive=d/'wsi.zip';root=d/'project'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('blca/tiles/features/pt_files/slide.pt',b'feature-bytes')
                z.writestr('raw.svs',b'not-needed')
            item=dict(name='blca.zip',kind='wsi',destination='WSIs/blca-003')
            r=extract_archive(archive,item,root,reserve_gb=0)
            self.assertEqual(len(r),1)
            self.assertEqual((root/'WSIs/blca-003/pt_files/slide.pt').read_bytes(),b'feature-bytes')
            extract_archive(archive,item,root,reserve_gb=0)
            (root/'WSIs/blca-003/pt_files/slide.pt').write_bytes(b'bad')
            with self.assertRaises(FileExistsError): extract_archive(archive,item,root,reserve_gb=0)
    def test_wrapper(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);archive=d/'meta.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('project/datasets_csv/metadata/tcga_blca.csv','a,b')
            extract_archive(archive,dict(name='m.zip',kind='tree',destination='datasets_csv'),d/'root',0)
            self.assertTrue((d/'root/datasets_csv/metadata/tcga_blca.csv').exists())
    def test_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);archive=d/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('../escape.pt','bad')
            with self.assertRaises(ValueError):extract_archive(archive,dict(name='bad.zip',kind='wsi',destination='WSIs/blca-003'),d/'root',0)

if __name__=='__main__':unittest.main()
