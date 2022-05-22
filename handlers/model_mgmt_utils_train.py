import os
import logging
import datetime
import shutil

import dtlpy as dl
from dtlpy.ml.train_utils import prepare_dataset

logger = logging.getLogger(name=__name__)


class ServiceRunner(dl.BaseServiceRunner):
    """
    Package runner class
    """

    def __init__(self):
        """

        """
        pass

    def train_on_snapshot(self,
                          snapshot: dl.Snapshot,
                          cleanup=False,
                          progress: dl.Progress = None,
                          context: dl.Context = None):
        # FROM PARENT
        """
            Train on existing snapshot.
            data will be taken from snapshot.datasetId
            configuration is as defined in snapshot.configuration
            upload the output the the snapshot's bucket (snapshot.bucket)
        """
        try:
            if isinstance(snapshot, str):
                snapshot = dl.snapshots.get(snapshot_id=snapshot)
            logger.info("Received {s} for training".format(s=snapshot.id))
            snapshot.status = 'training'
            if 'system' not in snapshot.metadata:
                snapshot.metadata['system'] = dict()
            snapshot.metadata['system']['trainExecutionId'] = context.execution_id
            snapshot.update()

            def on_epoch_end(epoch, n_epoch):
                if progress is not None:
                    progress.update(message='training epoch: {}/{}'.format(epoch, n_epoch), progress=epoch / n_epoch)

            model = snapshot.model

            logger.info("Building Model {n} ({i!r})".format(n=model.name, i=model.id))
            adapter = model.build()

            if snapshot is not None:
                logger.info("Loading Adapter with: {n} ({i!r})".format(n=snapshot.name, i=snapshot.id))
                logger.debug("Snapshot\n{}\n{}".format('=' * 8, snapshot.print(to_return=True)))
                adapter.load_from_snapshot(snapshot)

            root_path, data_path, output_path = adapter.prepare_training(root_path=os.path.join('tmp', snapshot.id))
            # Start the Train
            logger.info("Training {m_name!r} with snapshot {s_name!r} on data {d_path!r}".
                        format(m_name=adapter.model_name, s_name=snapshot.id, d_path=data_path))
            if progress is not None:
                progress.update(message='starting training')

            def on_epoch_end_callback(i_epoch, n_epoch):
                if progress is not None:
                    progress.update(progress=int(100 * (i_epoch + 1) / n_epoch),
                                    message='finished epoch: {}/{}'.format(i_epoch, n_epoch))

            adapter.train(data_path=data_path,
                          output_path=output_path,
                          on_epoch_end=on_epoch_end,
                          on_epoch_end_callback=on_epoch_end_callback)
            if progress is not None:
                progress.update(message='saving snapshot',
                                progress=99)

            adapter.save_to_snapshot(local_path=output_path, replace=True)

            ###########
            # cleanup #
            ###########
            if cleanup:
                shutil.rmtree(output_path, ignore_errors=True)
        except Exception:
            snapshot.status = 'training'
            snapshot.update()
            raise
        return adapter.snapshot.id

    def train_from_dataset(self,
                           from_snapshot: dl.Snapshot,
                           dataset: dl.Dataset,
                           filters=None,
                           # new training params
                           snapshot_name=None,
                           configuration=None,
                           progress: dl.Progress = None):
        """
            Create a cloned snapshot from dataset
            Train using new snapshot

        Args:
            from_snapshot (dl.Snapshot, optional): What is the `source` Snapshot to clone from
            dataset (dl.Dataset): source dataset
            filters (dl.Filters, optional): how to create the cloned dataset. Defaults to None.
            snapshot_name (str, optional): New cloned snapshot name. Defaults to None==> <model_name>-<dataset_name>-<YYMMDD-HHMMSS>.
            configuration (dict, optional): updated configuration in the cloned snapshot. Defaults to None.
            progress (dl.Progress, optional): [description]. Defaults to None.

        Returns:
            dl.Snapshot: Cloned snapshot
        """
        logger.info("Recieved a dataset {d!r} to train from".format(d=dataset.id))
        snapshot = self.clone_snapshot_from_dataset(
            dataset=dataset,
            filters=filters,
            from_snapshot=from_snapshot,
            snapshot_name=snapshot_name,
            configuration=configuration,
            progress=progress
        )

        return self.train_on_snapshot(snapshot=snapshot,
                                      progress=progress)

    def clone_snapshot_from_dataset(self,
                                    from_snapshot: dl.Snapshot,
                                    dataset: dl.Dataset,
                                    filters=None,
                                    # new training params
                                    snapshot_name=None,
                                    configuration=None,
                                    progress: dl.Progress = None):
        """Creates a new snapshot from dataset
            Functionality is split - for the use from UI
        

        Args:
            from_snapshot (dl.Snapshot, optional): What is the `source` Snapshot to clone from
            dataset (dl.Dataset): source dataset
            filters (dl.Filters, optional): how to create the cloned dataset. Defaults to None.
            snapshot_name (str, optional): New cloned snapshot name. Defaults to None==> <model_name>-<dataset_name>-<YYMMDD-HHMMSS>.
            configuration (dict, optional): updated configuration in the cloned snapshot. Defaults to None.
            progress (dl.Progress, optional): [description]. Defaults to None.

        Returns:
            dl.Snapshot: Cloned snapshot
        """
        logger.info("Recieved a dataset {d!r} to use in cloned version of orig snapshot {s!r} ".
                    format(d=dataset.id, s=from_snapshot.id))
        # Base entities
        model = from_snapshot.model
        project = dataset.project  # This is the destenation project

        if isinstance(filters, dict):
            t_filters = filters
            filters = dl.Filters()
            filters.custom_filter = t_filters
        if progress is not None:
            progress.update(message='preparing dataset', progress=5)

        partitions = {dl.SnapshotPartitionType.TRAIN: 0.8,
                      dl.SnapshotPartitionType.VALIDATION: 0.2}
        cloned_dataset = prepare_dataset(dataset,
                                         partitions=partitions,
                                         filters=filters)
        if snapshot_name is None:
            snapshot_name = '{}-{}-{}'.format(model.name,
                                              cloned_dataset.name,
                                              datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        if configuration is None:
            configuration = dict()

        if progress is not None:
            progress.update(message='creating snapshot', progress=10)

        bucket = project.buckets.create(bucket_type=dl.BucketType.ITEM,
                                        model_name=model.name,
                                        snapshot_name=snapshot_name)
        cloned_snapshot = from_snapshot.clone(snapshot_name=snapshot_name,
                                              configuration=configuration,
                                              bucket=bucket,
                                              project_id=project.id,
                                              dataset_id=cloned_dataset.id)
        return cloned_snapshot


def train_yolox_test(env='prod'):
    import logging
    logging.basicConfig(level='INFO')
    dl.setenv(env)
    project = dl.projects.get(project_name='COCO ors')
    model = project.models.get(model_name='YOLOX')
    ##############################
    ################## all should be in function
    # FIXME - package changed - we need to refactor code
    self = ServiceRunner()

    ########################
    #########################
    snapshot_name = 'second-fruit'
    # delete if exists
    try:
        snap = model.snapshots.get(snapshot_name=snapshot_name)
        snap.dataset.delete(sure=True, really=True)
        snap.delete()
    except Exception:
        pass

    dataset = project.datasets.get(dataset_name='FruitImage')
    configuration = {'batch_size': 2,
                     'start_epoch': 0,
                     'max_epoch': 5,
                     'input_size': (256, 256)}

    self.train_from_dataset(dataset=dataset,
                            filters=dict(),
                            snapshot_name=snapshot_name,
                            configuration=configuration)


def train_yolov5_test(env='rc'):
    # inputs
    import logging
    logging.basicConfig(level='INFO')
    dl.setenv(env)
    project_name = 'DataloopModels'
    model_name = 'yolo-v5'
    snapshot_name = 'pretrained-yolo-v5-small'

    project = dl.projects.get(project_name=project_name)
    model = project.models.get(model_name=model_name)
    pretrained_snapshot = model.snapshots.get(snapshot_name=snapshot_name)

    ####################
    # Destenation params
    ####################
    dst_project = dl.projects.get(project_name='roberto-sandbox')
    dataset = dst_project.datasets.get(dataset_name='ds-2-frozen')
    dst_snapshot_name = 'snap-utils-train'
    # delete if exists
    try:
        snap = dst_project.snapshots.get(snapshot_name=snapshot_name)
        print("Found snapshot - deleting dataset and snapshot")
        snap.dataset.delete(sure=True, really=True)
        snap.delete()
    except Exception:
        pass

    runner = ServiceRunner()

    configuration = {'batch_size': 2,
                     'num_epochs': 3,
                     }

    dst_snapshot = runner.train_from_dataset(
        from_snapshot=pretrained_snapshot,
        dataset=dataset,
        filters=dict(),
        snapshot_name=dst_snapshot_name,
        configuration=configuration
    )
    print("Returned snapshot")
    print(dst_snapshot.print(to_return=False))


if __name__ == '__main__':
    print("Test function")
    # train_yolox_test()
    train_yolov5_test()
