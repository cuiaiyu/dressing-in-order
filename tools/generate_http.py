import json, os

HEAD = ['<!DOCTYPE html>','<html>', '<body>']
TAIL = ['</body>', '</html>']




class HTMLGenerator:
    def __init__(self, img_dirs):
        self.img_dirs = img_dirs
    def _generate_head(self):
        out_string = '\n'.join(HEAD) + '\n'
        return out_string

    def _generate_tail(self):
        out_string = '\n'.join(TAIL) + '\n'
        return out_string
            

    def _generate_subject_section(self, exp_name='default', epoch=0, subset='Full'):
        out_string = '<h1> Visualization </h1>' + '\n'
        #out_string += '<h2>Exp Meta</h2>' + '\n'
        #out_string += '<p>Exp Name: %s</p>' % exp_name + '\n'
        #out_string += '<p>Epoch: %d</p>' % epoch + '\n'
        #out_string += '<p>Subset: %s </p>' % subset + '\n'
        return out_string

    def _generate_retrieval_table(self):
        # head
        out_string = '<h2>Retrieval Result</h2>' + '\n'
        out_string += '<table>' + '\n'
        out_string += '<tr>' + '\n'
        out_string += '<th>index</th>'  + '\n'
        for name in self.img_dirs:
            out_string += '<th>%s</th>' % name + '\n'
        out_string += '</tr>' + '\n'
        # body
        keys = list(self.img_dirs.keys())
        N = len(os.listdir(self.img_dirs[keys[0]]))
        for i in range(1,N+1):
            out_string += '<tr>' + '\n'
            out_string += '<td>index</td>'  + '\n'
            for name in self.img_dirs:
                out_string += '<td>%s</td>' % name + '\n'
            out_string += '</tr>' + '\n'
            out_string += '<tr>' + '\n'
            out_string += '<td>%d</td>' % i + '\n'
            fn = 'generated_%d.jpg' % i
            for folder in self.img_dirs:
                img_path = os.path.join(img_dirs[folder], fn)
                if folder == 'ADGAN':
                    out_string += '<td><img src="%s" height=256 width=%s" style="border:solid; border-color:red;"/></td>' % (img_path, 176 * 5) + '\n'
                else:
                    out_string += '<td><img src="%s" height=256 width=%s" style="border:solid; border-color:red;"/></td>' % (img_path, 176 * 3) + '\n'
            out_string += '</tr>' + '\n'

        return out_string

   
    def generate(self, out_path='tmp.html'):
        out_string = self._generate_head()
        #print(out_string)
        #subset_title = 'false_only_{} (if True, only those queries with R@10 = 0 displayed.)'
        out_string += self._generate_subject_section()
        out_string += self._generate_retrieval_table()
        out_string += self._generate_tail()
        with open(out_path, 'w') as f:
            f.write(out_string)


if __name__ == '__main__':
    if False:
        root = '../data/'
        img_dirs = {
            "ADGAN": "checkpoints/yifang_800",
            "GFLA": "checkpoints/eval_results_256jpg/fashion",
            "large, last week":"checkpoints/adseq2_vgg_large1_latest_jpg",
            "large, new":"checkpoints/adseq2_vgg_large_square_latest_jpg",
        }
        Generator = HTMLGenerator(img_dirs=img_dirs)
        Generator.generate()
    else:
        
        img_dirs = {
            "ADGANv": "checkpoints/viton_yifang_1000",
            "ours-large-176":"checkpoints/viton_adseq2_vgg_large1_latest",
        }
        Generator = HTMLGenerator(img_dirs=img_dirs)
        Generator.generate("viton.html")