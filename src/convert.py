import os
import shutil

import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import (
    file_exists,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from tqdm import tqdm

import src.settings as s


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # Possible structure for bbox case. Feel free to modify as you needs.

    train_path = "/home/alex/DATASETS/TODO/VisDrone/train/images"
    val_path = "/home/alex/DATASETS/TODO/VisDrone/val/images"
    test_dev_path = "/home/alex/DATASETS/TODO/VisDrone/test-dev/images"
    test_challenge_path = "/home/alex/DATASETS/TODO/VisDrone/test-challenge/images"

    batch_size = 30
    images_ext = ".jpg"
    anns_ext = ".txt"

    ds_name_to_data = {"train": train_path, "val": val_path, "test": test_dev_path}

    def create_ann(image_path):
        labels = []
        tags = []

        if ds_name == "test":
            if image_path.split("/")[-3] == "test-dev":
                test_tag = sly.Tag(dev)
                tags.append(test_tag)
            else:
                test_tag = sly.Tag(challenge)
                tags.append(test_tag)

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        ann_path = image_path.replace("images", "annotations").replace(".jpg", ".txt")

        if file_exists(ann_path):
            with open(ann_path) as f:
                content = f.read().split("\n")

                for curr_data in content:
                    if len(curr_data) != 0:
                        curr_data = curr_data.split(",")
                        l_tags = []
                        obj_class = idx_to_class[int(curr_data[5])]
                        occlusion_meta = idx_to_occlusion[int(curr_data[7])]
                        occlusion = sly.Tag(occlusion_meta)
                        l_tags.append(occlusion)

                        truncation_meta = idx_to_truncation[int(curr_data[6])]
                        truncation = sly.Tag(truncation_meta)
                        l_tags.append(truncation)

                        left = int(curr_data[0])
                        top = int(curr_data[1])
                        right = left + int(curr_data[2])
                        bottom = top + int(curr_data[3])
                        rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
                        label = sly.Label(rectangle, obj_class, tags=l_tags)
                        labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    idx_to_class = {
        0: sly.ObjClass("ignored region", sly.Rectangle),
        1: sly.ObjClass("pedestrian", sly.Rectangle),
        2: sly.ObjClass("people", sly.Rectangle),
        3: sly.ObjClass("bicycle", sly.Rectangle),
        4: sly.ObjClass("car", sly.Rectangle),
        5: sly.ObjClass("van", sly.Rectangle),
        6: sly.ObjClass("truck", sly.Rectangle),
        7: sly.ObjClass("tricycle", sly.Rectangle),
        8: sly.ObjClass("awning tricycle", sly.Rectangle),
        9: sly.ObjClass("bus", sly.Rectangle),
        10: sly.ObjClass("motor", sly.Rectangle),
        11: sly.ObjClass("other", sly.Rectangle),
    }

    challenge = sly.TagMeta("challenge", sly.TagValueType.NONE)
    dev = sly.TagMeta("dev", sly.TagValueType.NONE)
    no_occlusion = sly.TagMeta("no occlusion", sly.TagValueType.NONE)
    partial_occlusion = sly.TagMeta("partial occlusion", sly.TagValueType.NONE)
    heavy_occlusion = sly.TagMeta("heavy occlusion", sly.TagValueType.NONE)

    idx_to_occlusion = {0: no_occlusion, 1: partial_occlusion, 2: heavy_occlusion}

    no_truncation = sly.TagMeta("no truncation", sly.TagValueType.NONE)
    partial_truncation = sly.TagMeta("partial truncation", sly.TagValueType.NONE)

    idx_to_truncation = {0: no_truncation, 1: partial_truncation}

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=list(idx_to_class.values()),
        tag_metas=[
            challenge,
            dev,
            no_occlusion,
            partial_occlusion,
            heavy_occlusion,
            no_truncation,
            partial_truncation,
        ],
    )
    api.project.update_meta(project.id, meta.to_json())

    for ds_name, data_path in ds_name_to_data.items():

        dataset = api.dataset.create(
            project.id, get_file_name(ds_name), change_name_if_conflict=True
        )

        images_names = os.listdir(data_path)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for images_names_batch in sly.batched(images_names, batch_size=batch_size):
            img_pathes_batch = [
                os.path.join(data_path, image_name) for image_name in images_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(images_names_batch))

    images_names = os.listdir(test_challenge_path)

    progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

    for images_names_batch in sly.batched(images_names, batch_size=batch_size):
        img_pathes_batch = [
            os.path.join(data_path, image_name) for image_name in images_names_batch
        ]

        img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
        img_ids = [im_info.id for im_info in img_infos]

        anns = [create_ann(image_path) for image_path in img_pathes_batch]
        api.annotation.upload_anns(img_ids, anns)

        progress.iters_done_report(len(images_names_batch))

    return project
