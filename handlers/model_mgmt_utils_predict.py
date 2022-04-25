import logging
import dtlpy as dl

logger = logging.getLogger(name=__name__)


class ServiceRunner(dl.BaseServiceRunner):
    """
    Package runner class
    """

    def __init__(self,
                 project_id=None, project_name=None,
                 model_id=None, model_name=None,
                 snapshot_name=None, snapshot_id=None):
        """

        """
        # get elements
        self.project = dl.projects.get(project_name=project_name, project_id=project_id)

        try:
            model = self.project.models.get(model_name=model_name, model_id=model_id)
        except dl.exceptions.NotFound:
            logger.debug("Model not found in project")
            model = dl.models.get(model_name=model_name, model_id=model_id)

        if snapshot_id is not None or snapshot_name is not None:
            snapshot = model.snapshots.get(snapshot_id=snapshot_id, snapshot_name=snapshot_name)
        else:
            snapshot = None

        logger.info("Building Model {n} ({i!r})".format(n=model.name, i=model.id))
        adapter = model.build()

        if snapshot is not None:
            logger.info("Loading Adapter with: {n} ({i!r})".format(n=snapshot.name, i=snapshot.id))
            logger.debug("Snapshot\n{}\n{}".format('=' * 8, snapshot.print(to_return=True)))
            adapter.load_from_snapshot(snapshot)
        self.adapter = adapter

    def predict_item(self, item: dl.Item, with_upload=True, with_return=False):
        item_mime_type = item.mimetype.split('/')[0]
        model = self.adapter.model_entity

        if item_mime_type != model.input_type:
            raise ValueError("Trying to predict item of type {item_t} While the model works on {model_t}".
                             format(item_t=item_mime_type, model_t=model.input_type))

        predictions = self.adapter.predict_items(items=[item],
                                                 with_upload=with_upload)
        if with_return:
            return predictions


def test_yolov5_predict(env='prod', item_id=None):
    dl.setenv(env)
    project_name = 'DataloopModels'
    model_name = 'yolo-v5'
    snapshot_name = 'pretrained-yolo-v5-small'

    project = dl.projects.get(project_name=project_name)
    model = project.models.get(model_name=model_name)

    try:
        pretrained_snapshot = project.snapshots.get(snapshot_name=snapshot_name)
    except dl.exceptions.TokenExpired as err:
        print("Please login to environment {!r}".format(dl.environment()))
        raise err
    snapshot_id = pretrained_snapshot.id

    runner = ServiceRunner(project_id=project.id,
                           model_id=model.id,
                           snapshot_id=snapshot_id)

    # PREDICT
    if item_id is None:
        # item = dl.items.get(item_id='60b889904c3fa7463d010348')   # in coco-sample ds
        item = dl.items.get(item_id='6013173401c9e7ff2f6b4827')  # in ds-1-frozen
    else:
        item = dl.items.get(item_id=item_id)

    print("prediction item : {!r}".format(item.id))

    predictions = runner.predict_item(item=item, with_upload=False, with_return=True)
    # print(item.annotations.list())
    print(predictions)


if __name__ == '__main__':
    print("Train test")
    test_yolov5_predict()
