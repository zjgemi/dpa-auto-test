from dflow import argo_enumerate, S3Artifact, Step, upload_artifact, Workflow
from dflow.plugins.dispatcher import DispatcherExecutor
from dflow.python import Artifact, OP, PythonOPTemplate, Slices
import json
import os
from pathlib import Path
import sys
from typing import List


def get_artifact(data, name="data"):
    if isinstance(data, str) and data.startswith("oss://"):
        return S3Artifact(key=data[6:])
    art = upload_artifact(data)
    if hasattr(art, "key"):
        print("%s has been uploaded to oss://%s" % (name, art.key))
    return art


@OP.function
def run_dp_train(
    dataset_name: str,
    script: Artifact(Path),
    training_data: Artifact(List[Path]),
    validation_data: Artifact(List[Path]),
) -> {
    "log": Artifact(Path),
    "lcurve": Artifact(Path),
}:
    with open(script, "r") as f:
        config = json.load(f)
    config["training"]["training_data"]["systems"] = [str(p) for p in training_data]
    config["training"]["validation_data"]["systems"] = [str(p) for p in validation_data]
    with open("input.json", "w") as f:
        json.dump(config, f, indent=4)

    # os.system("dp --pt train input.json")
    with open("log", "w") as f:
        f.write("This is log for %s" % dataset_name)
    with open("lcurve.out", "w") as f:
        f.write("This is lcurve for %s" % dataset_name)
    return {
        "log": Path("log"),
        "lcurve": Path("lcurve.out"),
    }


@OP.function
def summary(
    names: List[str],
    logs: Artifact(List[Path]),
    lcurves: Artifact(List[Path]),
) -> {}:
    print(names)
    print(logs)
    print(lcurves)
    return {}


def main(config):
    wf = Workflow(name=config.get("name", "dpa-test"))
    train_config = config["train"]
    train_script = train_config["template_script"]
    data_dict = train_config["data_dict"]

    name_list = []
    train_art_list = []
    valid_art_list = []
    for name, dataset in data_dict.items():
        name_list.append(name)
        train_art_list.append(get_artifact(dataset["train"], "Training data of " + name))
        valid_art_list.append(get_artifact(dataset["valid"], "Validation data of " + name))

    train_step = Step(
        name="train",
        template=PythonOPTemplate(
            run_dp_train,
            image=train_config["image"],
            slices=Slices(
                "{{item.order}}",
                input_artifact=["training_data", "validation_data"],
                output_artifact=["log", "lcurve"],
                sub_path=True,
                create_dir=True,
            ),
        ),
        parameters={
            "dataset_name": "{{item.name}}",
        },
        artifacts={
            "script": upload_artifact(train_script),
            "training_data": train_art_list,
            "validation_data": valid_art_list,
        },
        key="train-{{item.name}}",
        executor=DispatcherExecutor(**train_config["executor"]) if train_config.get("executor") else None,
        with_param=argo_enumerate(name=name_list),
    )
    wf.add(train_step)

    summary_config = config["summary"]
    summary_step = Step(
        name="summary",
        template=PythonOPTemplate(
            summary,
            image=summary_config["image"],
        ),
        parameters={
            "names": name_list,
        },
        artifacts={
            "logs": train_step.outputs.artifacts["log"],
            "lcurves": train_step.outputs.artifacts["lcurve"],
        },
        key="summary",
        executor=DispatcherExecutor(**summary_config["executor"]) if summary_config.get("executor") else None,
    )
    wf.add(summary_step)
    wf.submit()


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        config = json.load(f)
    main(config)
