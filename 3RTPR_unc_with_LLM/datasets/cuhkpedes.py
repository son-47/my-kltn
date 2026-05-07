import os.path as op
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset


class CUHKPEDES(BaseDataset):
    """
    CUHK-PEDES

    Reference:
    Person Search With Natural Language Description (CVPR 2017)

    URL: https://openaccess.thecvf.com/content_cvpr_2017/html/Li_Person_Search_With_CVPR_2017_paper.html

    Dataset statistics:
    ### identities: 13003
    ### images: 40206,  (train)  (test)  (val)
    ### captions:
    ### 9 images have more than 2 captions
    ### 4 identity have only one image

    annotation format:
    [{'split', str,
      'captions', list,
      'file_path', str,
      'processed_tokens', list,
      'id', int}...]
    """
    dataset_dir = 'CUHK-PEDES'
    # Alternative annotation paths (for LLM-DA++ processed annotations with captions_bt)
    alt_annotation_subdir = 'annotation'

    def __init__(
        self,
        root='',
        verbose=True,
        use_augmented=False,
        annotation_dir=None,
        annotation_source="auto",
        annotation_train_source=None,
        annotation_eval_source=None,
        annotation_train_dir=None,
        annotation_eval_dir=None,
    ):
        super(CUHKPEDES, self).__init__()
        self.name = "CUHK-PEDES"
        dataset_dir_name = self.dataset_dir
        self.dataset_dir = op.join(root, dataset_dir_name)
        self.img_dir = op.join(self.dataset_dir, 'imgs/')

        # Determine which annotation path to use
        # annotation_source: auto | dir | raw
        annotation_source = annotation_source or "auto"
        train_source = annotation_train_source or annotation_source
        eval_source = annotation_eval_source or annotation_source
        train_dir = annotation_train_dir or annotation_dir
        eval_dir = annotation_eval_dir or annotation_dir

        def resolve_anno_path(source, anno_dir):
            source = source or "auto"
            if source == "raw":
                return op.join(self.dataset_dir, 'reid_raw.json')
            if source == "dir":
                if anno_dir:
                    return op.join(anno_dir, dataset_dir_name)
                return op.join(root, self.alt_annotation_subdir, dataset_dir_name)
            # auto
            if anno_dir:
                return op.join(anno_dir, dataset_dir_name)
            alt_anno_path = op.join(root, self.alt_annotation_subdir, dataset_dir_name)
            if op.exists(alt_anno_path):
                return alt_anno_path
            return op.join(self.dataset_dir, 'reid_raw.json')

        self.train_anno_path = resolve_anno_path(train_source, train_dir)
        self.eval_anno_path = resolve_anno_path(eval_source, eval_dir)
        self.anno_path = self.eval_anno_path

        self.logger.info(
            "Dataset %s: root=%s img_dir=%s train_anno_path=%s eval_anno_path=%s use_augmented=%s train_annotation_source=%s eval_annotation_source=%s",
            self.name,
            root,
            self.img_dir,
            self.train_anno_path,
            self.eval_anno_path,
            use_augmented,
            train_source,
            eval_source,
        )

        self._check_before_run()

        # Train annotations
        if op.isdir(self.train_anno_path):
            self.train_annos = read_json(op.join(self.train_anno_path, 'train_reid.json'))
            self.logger.info(
                "Train annotation source: directory (train_reid.json)"
            )
        else:
            self.train_annos, _, _ = self._split_anno(self.train_anno_path)
            self.logger.info(
                "Train annotation source: combined file (split by 'split' field)"
            )

        # Eval annotations (val/test)
        if op.isdir(self.eval_anno_path):
            self.val_annos = read_json(op.join(self.eval_anno_path, 'val_reid.json'))
            self.test_annos = read_json(op.join(self.eval_anno_path, 'test_reid.json'))
            self.logger.info(
                "Eval annotation source: directory (val_reid.json/test_reid.json)"
            )
        else:
            _, self.test_annos, self.val_annos = self._split_anno(self.eval_anno_path)
            self.logger.info(
                "Eval annotation source: combined file (split by 'split' field)"
            )

        self.logger.info(
            "Annotation counts: train=%d val=%d test=%d",
            len(self.train_annos),
            len(self.val_annos),
            len(self.test_annos),
        )
        if len(self.val_annos) == 0:
            self.logger.info(
                "Validation split is empty; val_dataset=val will yield an empty loader"
            )

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True, use_augmented=use_augmented)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)
        self.avg_len = sum([len(cap[-1]) for cap in self.train]) / len(self.train)

        if verbose:
            self.logger.info("=> CUHK-PEDES Images and Captions are loaded")
            self.show_dataset_info()


    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno['split'] == 'train':
                train_annos.append(anno)
            elif anno['split'] == 'test':
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos

  
    def _process_anno(self, annos: List[dict], training=False, use_augmented: bool = False):
        pid_container = set()
        if training:
            dataset = []
            image_id = 0
            for anno in annos:
                pid = int(anno['id']) - 1 # make pid begin from 0
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['file_path'])
                captions = anno['captions'] # caption list
                # Augmented captions from LLM-DA++ (stored in captions_bt field)
                augmented_captions = anno.get('captions_bt', [''] * len(captions))
                # Ensure same length as captions
                if len(augmented_captions) != len(captions):
                    augmented_captions = [''] * len(captions)
                for idx, caption in enumerate(captions):
                    aug_caption = augmented_captions[idx] if idx < len(augmented_captions) else ''
                    if use_augmented and aug_caption:
                        dataset.append((pid, image_id, img_path, caption, aug_caption))
                    else:
                        dataset.append((pid, image_id, img_path, caption, aug_caption))
                image_id += 1
            for idx, pid in enumerate(pid_container):
                # check pid begin from 0 and no break
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['file_path'])
                img_paths.append(img_path)
                image_pids.append(pid)
                caption_list = anno['captions'] # caption list
                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(pid)
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        for path in [self.train_anno_path, self.eval_anno_path]:
            if not op.isdir(path) and not op.exists(path):
                raise RuntimeError("'{}' is not available".format(path))