import os.path as op
from typing import List

from utils import read_json
from .bases import BaseDataset


class RSTPReid(BaseDataset):
    """
    RSTPReid

    Reference:
    DSSL: Deep Surroundings-person Separation Learning for Text-based Person Retrieval MM 21

    URL: http://arxiv.org/abs/2109.05534

    Dataset statistics:
    # identities: 4101
    """
    dataset_dir = 'RSTPReid'
    alt_annotation_subdir = 'annotation'

    def __init__(self, root='', verbose=True, use_augmented=False, annotation_dir=None, annotation_source="auto"):
        super(RSTPReid, self).__init__()
        self.name = "RSTPReid"
        dataset_dir_name = self.dataset_dir
        self.dataset_dir = op.join(root, dataset_dir_name)
        self.img_dir = op.join(self.dataset_dir, 'imgs/')

        # Determine which annotation path to use
        # annotation_source: auto | dir | raw
        annotation_source = annotation_source or "auto"
        if annotation_source == "raw":
            self.anno_path = op.join(self.dataset_dir, 'data_captions.json')
        elif annotation_source == "dir":
            if annotation_dir:
                self.anno_path = op.join(annotation_dir, dataset_dir_name)
            else:
                self.anno_path = op.join(root, self.alt_annotation_subdir, dataset_dir_name)
        else:
            if annotation_dir:
                self.anno_path = op.join(annotation_dir, dataset_dir_name)
            else:
                alt_anno_path = op.join(root, self.alt_annotation_subdir, dataset_dir_name)
                if op.exists(alt_anno_path):
                    self.anno_path = alt_anno_path
                else:
                    self.anno_path = op.join(self.dataset_dir, 'data_captions.json')

        self.logger.info(
            "Dataset %s: root=%s img_dir=%s anno_path=%s use_augmented=%s annotation_source=%s",
            self.name,
            root,
            self.img_dir,
            self.anno_path,
            use_augmented,
            annotation_source,
        )
        self._check_before_run()

        # Check if using split annotation files (annotation folder) or combined file
        if op.isdir(self.anno_path):
            self.train_annos = read_json(op.join(self.anno_path, 'train_reid.json'))
            self.val_annos = read_json(op.join(self.anno_path, 'val_reid.json'))
            self.test_annos = read_json(op.join(self.anno_path, 'test_reid.json'))
            self.logger.info(
                "Annotation source: directory (train_reid.json/val_reid.json/test_reid.json)"
            )
        else:
            self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)
            self.logger.info(
                "Annotation source: combined file (split by 'split' field)"
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
            self.logger.info("=> RSTPReid Images and Captions are loaded")
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
                pid = int(anno['id'])
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['img_path'])
                captions = anno['captions'] # caption list
                # Augmented captions from LLM-DA++ (stored in captions_bt field)
                augmented_captions = anno.get('captions_bt', [''] * len(captions))
                # Ensure same length as captions
                if len(augmented_captions) != len(captions):
                    augmented_captions = [''] * len(captions)
                for idx, caption in enumerate(captions):
                    aug_caption = augmented_captions[idx] if idx < len(augmented_captions) else ''
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
                img_path = op.join(self.img_dir, anno['img_path'])
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
        if not op.isdir(self.anno_path) and not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
