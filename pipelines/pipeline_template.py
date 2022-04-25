import json
import dtlpy as dl


def replace(to_replace, replace_values):
    if isinstance(to_replace, dict):
        for key, value in to_replace.items():
            to_replace[key] = replace(value, replace_values)
    elif isinstance(to_replace, list):
        for key, value in enumerate(to_replace):
            to_replace[key] = replace(value, replace_values)
    elif isinstance(to_replace, str):
        if to_replace.startswith('$'):
            param = to_replace[1:]
            if param not in replace_values:
                raise ValueError('missing {} in replace dict'.format(param))
            return replace_values[param]
    return to_replace


def main():
    dl.setenv('rc')
    with open('template_pipelines.json', 'r') as f:
        pipelines_json = json.load(f)

    deploy_project = dl.projects.get('Model Management App')
    package_project = dl.projects.get('Model Management App')
    replace_values = {'projectId': deploy_project.id,
                      'orgId': deploy_project.org['id'],
                      'funcProjectName': package_project.name,
                      'funcProjectId': package_project.id,
                      'funcPackageName': 'model-mgmt-app',
                      'funcModuleName': 'model-mgmt-app-train',
                      'funcFunctionName': 'train_on_snapshot',
                      'funcServiceName': 'model-mgmt-app-train',
                      }

    ready_pipeline = replace(pipelines_json, replace_values)
    pipeline = deploy_project.pipelines.create(pipeline_json=ready_pipeline)
    pipeline.install()
