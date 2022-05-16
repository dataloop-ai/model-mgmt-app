import os

import dtlpy as dl

project_name = 'Model Management App'
dl.setenv('rc')
project = dl.projects.get(project_name=project_name)
package_name = 'model-mgmt-app'

###########
# Modules #
###########
module_train = dl.PackageModule(
    entry_point='handlers/model_mgmt_utils_train.py',
    name=package_name + '-train',
    class_name='ServiceRunner',
    init_inputs=[],
    functions=[
        dl.PackageFunction(
            name='train_on_snapshot',
            display_name='train-on-snapshot-{}'.format(package_name),
            description="run train of the model on the specific snapshot id",
            inputs=[
                dl.FunctionIO(type=dl.PackageInputType.SNAPSHOT, name='snapshot'),
                dl.FunctionIO(type=dl.PackageInputType.JSON, name='cleanup'),
            ],
            outputs=[dl.FunctionIO(type=dl.PackageInputType.SNAPSHOT, name='snapshot')]
        ),
        dl.PackageFunction(
            name='train_from_dataset',
            display_name='train-on-dataset-{}'.format(package_name),
            description="run train of a model on a raw dataset",
            inputs=[
                dl.FunctionIO(type=dl.PackageInputType.DATASET, name='dataset'),
                dl.FunctionIO(type=dl.PackageInputType.JSON, name='filters'),
                dl.FunctionIO(type=dl.PackageInputType.SNAPSHOT, name='from_snapshot'),
                dl.FunctionIO(type=dl.PackageInputType.JSON, name='snapshot_name'),
                dl.FunctionIO(type=dl.PackageInputType.JSON, name='configuration'),
            ], outputs=[dl.FunctionIO(type=dl.PackageInputType.JSON, name='snapshot_id')]
        ),
        dl.PackageFunction(
            name='clone_snapshot_from_dataset',
            display_name='clone-from-dataset-{}'.format(package_name),
            description="clones the current scope to a new snapshot",
            inputs=[
                dl.FunctionIO(type=dl.PackageInputType.DATASET, name='dataset'),
                dl.FunctionIO(type=dl.PackageInputType.JSON, name='filters'),
                dl.FunctionIO(type=dl.PackageInputType.SNAPSHOT, name='from_snapshot'),
                dl.FunctionIO(type=dl.PackageInputType.JSON, name='snapshot_name'),
                dl.FunctionIO(type=dl.PackageInputType.JSON, name='configuration'),
            ], outputs=[dl.FunctionIO(type=dl.PackageInputType.JSON, name='snapshot_id')]
        )
    ])

module_predict = dl.PackageModule(
    entry_point='handlers/model_mgmt_utils_predict.py',
    name=package_name + '-predict',
    class_name='ServiceRunner',
    init_inputs=[
        dl.FunctionIO(type="Json", name='project_name'),
        dl.FunctionIO(type="Json", name='project_id'),
        dl.FunctionIO(type="Json", name='model_name'),
        dl.FunctionIO(type="Json", name='model_id'),
        dl.FunctionIO(type="Json", name='snapshot_name'),
        dl.FunctionIO(type="Json", name='snapshot_id'),
    ],
    functions=[
        dl.PackageFunction(
            name='predict_item',
            display_name='predict-item-{}'.format(package_name),
            description='predict a single item with the object adapter',
            inputs=[
                dl.FunctionIO(type=dl.PackageInputType.ITEM, name='item'),
                dl.FunctionIO(type=dl.PackageInputType.JSON, name='with_upload'),
                dl.FunctionIO(type=dl.PackageInputType.JSON, name='with_return'),
            ],
            outputs=[dl.FunctionIO(type=dl.PackageInputType.ITEM, name='item')],
        )
    ])

#########
# Slots #
#########
slots = [
    dl.PackageSlot(
        module_name=package_name + '-train',
        function_name='clone_snapshot_from_dataset',
        display_name='Make Snapshot',
        post_action=dl.SlotPostAction(type=dl.SlotPostActionType.NO_ACTION),
        display_scopes=[
            dl.SlotDisplayScope(
                resource=dl.SlotDisplayScopeResource.DATASET_QUERY,
                filters={}
            )
        ]
    ),
    dl.PackageSlot(
        module_name=package_name + '-predict',
        function_name='predict_item',
        display_name='PREDICT - TEST ME',
        post_action=dl.SlotPostAction(type=dl.SlotPostActionType.NO_ACTION),
        display_scopes=[
            dl.SlotDisplayScope(
                resource=dl.SlotDisplayScopeResource.ITEM,
                filters={}
            )
        ]
    ),
]

################
# push package #
################
codebase = dl.GitCodebase(git_url='https://github.com/dataloop-ai/model-mgmt-app.git',
                          git_tag='main')
package = project.packages.push(package_name=package_name,
                                modules=[module_train, module_predict],
                                slots=slots,
                                src_path=os.getcwd(),
                                codebase=codebase,
                                is_global=True
                                )

package = project.packages.get(package_name=package_name)

##################
# create service #
##################
#
# service = package.services.deploy(service_name=package_name + '-train',
#                                   module_name=package_name + '-train',
#                                   bot=bot_name,
#                                   init_input={},
#                                   runtime=dl.KubernetesRuntime(
#                                       pod_type=dl.InstanceCatalog.REGULAR_S,
#                                       runner_image='gcr.io/viewo-g/piper/agent/cpu/roberto_utils:2',
#                                       concurrency=1,
#                                       autoscaler=dl.KubernetesRabbitmqAutoscaler(min_replicas=1)
#                                       # FIXME - Always up
#                                   ),
#                                   )
#     service = package.services.deploy(service_name=package_name+'-predict',
#                                     module_name=package_name+'-predict',
#                                     bot=bot_name,
#                                     init_input={'project_name': project_name,
#                                                 'model_name': args.model_name,
#                                                 'snapshot_id': snapshot_id},
#                                     runtime=dl.KubernetesRuntime(
#                                         pod_type=dl.InstanceCatalog.REGULAR_S,
#                                         concurrency=1,
#                                         autoscaler=dl.KubernetesRabbitmqAutoscaler(min_replicas=1)  # FIXME - Always up
#                                     ),
#                                     )



##############
# EXECUTIONS #
##############
# if False:
#     config = dict()
#     for idx in range(0, nof_args, 2):
#         config[args.exe[idx]] = args.exe[idx + 1]
#
#     print(config)
#     print("waiting to make sure new snapshot")
#     time.sleep(8)
#
#     exe_in = dl.FunctionIO(type=dl.PackageInputType.JSON, name="config", value=config)
#     execution = service.execute(project_id=project.id,
#                                 function_name='execution_wrapper',
#                                 execution_input=[exe_in])
#
#     print("Execution {e_id!r} was sent to pkg {p} Service {s_id} version {s_v}} config{c}".
#           format(e_id=execution.id, p=package_name, s_id=service.id, s_v=service.version, c=config))
