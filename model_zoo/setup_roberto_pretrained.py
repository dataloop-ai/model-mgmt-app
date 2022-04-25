import dtlpy as dl

GITS = {
    'v3': 'https://github.com/dataloop-ai/keras_yolo3.git',
    'v5': 'https://github.com/dataloop-ai/yolov5.git'
}
GCS = {
    'v3': 'yolov3_keras/ipm/oct-v1-4-classes-large',
    'v5': 'yolov5_torch'
}
LOCALS = {
    'v3': '$ZOO_CONFIGS/yolov3_keras/base',
    'v5': '$ZOO_CONFIGS/yolov5_torch/base/yolov5x.pt',
}

dl.setenv('rc')
yolo_gen = 'v3'  # v5

project = dl.projects.get('DataloopModels')
sample_ds = project.datasets.get('coco-sample')

model_name = 'yolo-{}-github'.format(yolo_gen)
snapshot_name = '{}-pretrained'.format(yolo_gen)

# MODEL
try:
    model = project.models.get(model_name=model_name)
    print("found exiting model")
except dl.exceptions.NotFound as err:
    model = project.models.create(model_name=model_name,
                                  description='yolo {} with a git codebase'.format(yolo_gen),
                                  output_type='box',
                                  codebase=dl.GitCodebase(git_url=GITS.get(yolo_gen),
                                                          git_tag='Not-used'),
                                  entry_point='model_adapter.py' if yolo_gen == 'v5' else 'yolov3_adapter.py',
                                  )
    print("created new model - {}".format(model_name))
# model.print()

# SNAPSHOT
try:
    snapshot = model.snapshots.get(snapshot_name=snapshot_name)
    print("found exiting snapshot")
except dl.exceptions.NotFound as err:
    snapshot_ont = sample_ds.ontologies.get(ontology_id=sample_ds.ontology_ids[0])
    my_gcs_bucket = model.buckets.create(bucket_type=dl.BucketType.GCS,
                                         gcs_project_name='viewo',
                                         gcs_bucket_name='model_zoo_weights',
                                         prefix='/' + GCS.get(yolo_gen),
                                         )
    my_local_bucket = model.buckets.create(bucket_type=dl.BucketType.LOCAL,
                                           local_path=LOCALS.get(yolo_gen))

    snapshot = model.snapshots.create(snapshot_name=snapshot_name,
                                      dataset_id=sample_ds.id,
                                      ontology_id=sample_ds.ontology_ids[0],
                                      description="pretrained weights - saved at dataloop gcs",
                                      bucket=my_local_bucket,
                                      # configuration={'weights_filename': 'yolo.h5', 'input_shape': (800, 800)},
                                      )
    print("created new snapshot")

snapshot.print()
