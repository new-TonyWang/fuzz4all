"""Main script to run the fuzzing process."""

import os
import time

import click
from rich.traceback import install
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
import os
import time
install()

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from Fuzz4All.make_target import make_target_with_config
from Fuzz4All.target.target import Target
from Fuzz4All.util.util import load_config_file


def is_ollama_model(model_name):
    return model_name.startswith("ollama/") or model_name in ["llama2", "starcoder"]


def write_to_file(fo, file_name):
    try:
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(fo)
    except:
        pass


def fuzz(
    target: Target,
    number_of_iterations: int,
    total_time: int,
    output_folder: str,
    resume: bool,
    otf: bool,
):
    target.initialize()
    with Progress(
        TextColumn("Fuzzing • [progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ) as p:
        task = p.add_task("Fuzzing", total=number_of_iterations)
        count = 0
        start_time = time.time()

        if resume:
            n_existing = [
                int(f.split(".")[0])
                for f in os.listdir(output_folder)
                if f.endswith(".fuzz")
            ]
            n_existing.sort(reverse=True)
            if len(n_existing) > 0:
                count = n_existing[0] + 1
            log = f" (resuming from {count})"
            p.console.print(log)

        p.update(task, advance=count)

        while (
            count < number_of_iterations
            and time.time() - start_time < total_time * 3600
        ):
            fos = target.generate()
            if not fos:
                target.initialize()
                continue
            prev = []
            for index, fo in enumerate(fos):
                file_name = os.path.join(output_folder, f"{count}.fuzz")
                write_to_file(fo, file_name)
                count += 1
                p.update(task, advance=1)
                # validation on the fly
                if otf:
                    f_result, message = target.validate_individual(file_name)
                    target.parse_validation_message(f_result, message, file_name)
                    prev.append((f_result, fo))
            target.update(prev=prev)

def fuzz_iteration(target, count, output_folder, otf):
    fos = target.generate()
    if not fos:
        target.initialize()
        return count, []  # 跳过
    
    prev = []
    for fo in fos:
        file_name = os.path.join(output_folder, f"{count}.fuzz")
        write_to_file(fo, file_name)
        if otf:
            f_result, message = target.validate_individual(file_name)
            target.parse_validation_message(f_result, message, file_name)
            prev.append((f_result, fo))
        count += 1
    return count, prev


def run_fuzzing_async(
    target: Target,
    number_of_iterations: int,
    total_time: int,
    output_folder: str,
    resume: bool,
    otf: bool,
    workers=16
):
    target.initialize()
    with Progress(
        TextColumn("Fuzzing • [progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ) as p:
        task = p.add_task("Fuzzing", total=number_of_iterations)
        count = 0
        start_time = time.time()

        if resume:
            n_existing = [
                int(f.split(".")[0])
                for f in os.listdir(output_folder)
                if f.endswith(".fuzz")
            ]
            n_existing.sort(reverse=True)
            if len(n_existing) > 0:
                count = n_existing[0] + 1
            log = f" (resuming from {count})"
            p.console.print(log)

        p.update(task, advance=count)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            while (
                count < number_of_iterations
                and time.time() - start_time < total_time * 3600
            ):
                # 提交任务
                futures.append(executor.submit(fuzz_iteration, target, count, output_folder, otf))
                count += 1  # 提交计数增加

                # 控制 futures 数量，避免爆内存
                if len(futures) >= workers:
                    for future in as_completed(futures):
                        new_count, prev = future.result()
                        if prev:
                            target.update(prev=prev)
                        p.update(task, advance=1)  # 完成一个迭代再更新进度
                        futures.remove(future)
                        break

# evaluate against the oracle to discover any potential bugs
# used after the generation
def evaluate_all(target: Target):
    target.validate_all()


@click.group()
@click.option(
    "config_file",
    "--config",
    type=str,
    default=None,
    help="Path to the configuration file.",
)
@click.pass_context
def cli(ctx, config_file):
    """Run the main using a configuration file."""
    if config_file is not None:
        config_dict = load_config_file(config_file)
        ctx.ensure_object(dict)
        ctx.obj["CONFIG_DICT"] = config_dict


@cli.command("main_with_config")
@click.pass_context
@click.option(
    "folder",
    "--folder",
    type=str,
    default="Results/test",
    help="folder to store results",
)
@click.option(
    "cpu",
    "--cpu",
    is_flag=True,
    help="to use cpu",  # this is for GPU resource low situations where only cpu is available
)
@click.option(
    "batch_size",
    "--batch_size",
    type=int,
    default=30,
    help="batch size for the model",
)
@click.option(
    "model_name",
    "--model_name",
    type=str,
    default="bigcode/starcoderbase",
    help="model to use",
)
@click.option(
    "target",
    "--target",
    type=str,
    default="",
    help="specific target to run",
)
@click.option(
    "use_vllm_server",
    "--use_vllm_server",
    type=bool,
    is_flag=True,
    help="specific target to run",
)


def main_with_config(ctx, folder, cpu, batch_size, target, model_name, use_vllm_server=False):
    """Run the main using a configuration file."""
    config_dict = ctx.obj["CONFIG_DICT"]
    fuzzing = config_dict["fuzzing"]
    config_dict["fuzzing"]["output_folder"] = folder
    if cpu:
        config_dict["llm"]["device"] = "cpu"
    if batch_size:
        config_dict["llm"]["batch_size"] = batch_size
    if model_name != "":
        config_dict["llm"]["model_name"] = model_name
    if target != "":
        config_dict["fuzzing"]["target_name"] = target
    if not use_vllm_server:
        if config_dict.get("vllm_server_config", None) is not None:
            del config_dict["vllm_server_config"]
    else:
        if config_dict.get("vllm_server_config", None) is None:
            raise ValueError("vllm_server_config must be provided when use_vllm_server is True")
    # if vllm_model_name != "":
    #     config_dict["vllm_server_config"]["vllm_model_name"] = vllm_model_name
    #     config_dict["vllm_server_config"]["api_url"] = api_url
    #     config_dict["vllm_server_config"]["api_key"] = api_key
    print(config_dict)

    target = make_target_with_config(config_dict)
    if not fuzzing["evaluate"]:
        assert (
            not os.path.exists(folder) or fuzzing["resume"]
        ), f"{folder} already exists!"
        os.makedirs(fuzzing["output_folder"], exist_ok=True)
        fuzzing_args = {
            "target": target,
            "number_of_iterations": fuzzing["num"],
            "total_time": fuzzing["total_time"],
            "output_folder": folder,
            "resume": fuzzing["resume"],
            "otf": fuzzing["otf"],
        }
        if use_vllm_server:
            fuzzing_args["workers"] = config_dict["vllm_server_config"].get("workers", 16)                
            run_fuzzing_async(
                **fuzzing_args
            )
        else:
            fuzz(**fuzzing_args)
    else:
        evaluate_all(target)


if __name__ == "__main__":
    cli()
