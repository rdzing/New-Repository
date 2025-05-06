"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""
import csv

import os
from csv import QUOTE_NONNUMERIC
from pathlib import Path
from time import time

import cv2
import pandas as pd
from rich.table import Table

from src import constants
from src.defaults import CONFIG_DEFAULTS
from src.evaluation import EvaluationConfig, evaluate_concatenated_response
from src.logger import console, logger
from src.template import Template
from src.utils.file import Paths, setup_dirs_for_paths, setup_outputs_for_template
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils, Stats
from src.utils.parsing import get_concatenated_response, open_config_with_defaults

# Load processors
STATS = Stats()


def entry_point(input_dir, args):
    if not os.path.exists(input_dir):
        raise Exception(f"Given input directory does not exist: '{input_dir}'")
    curr_dir = input_dir
    return process_dir(input_dir, curr_dir, args)


def print_config_summary(
    curr_dir,
    omr_files,
    template,
    tuning_config,
    local_config_path,
    evaluation_config,
    args,
):
    logger.info("")
    table = Table(title="Current Configurations", show_header=False, show_lines=False)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_row("Directory Path", f"{curr_dir}")
    table.add_row("Count of Images", f"{len(omr_files)}")
    table.add_row("Set Layout Mode ", "ON" if args["setLayout"] else "OFF")
    table.add_row(
        "Markers Detection",
        "ON" if "CropOnMarkers" in template.pre_processors else "OFF",
    )
    table.add_row("Auto Alignment", f"{tuning_config.alignment_params.auto_align}")
    table.add_row("Detected Template Path", f"{template}")
    if local_config_path:
        table.add_row("Detected Local Config", f"{local_config_path}")
    if evaluation_config:
        table.add_row("Detected Evaluation Config", f"{evaluation_config}")

    table.add_row(
        "Detected pre-processors",
        f"{[pp.__class__.__name__ for pp in template.pre_processors]}",
    )
    console.print(table, justify="center")


def process_dir(
    root_dir,
    curr_dir,
    args,
    template=None,
    tuning_config=CONFIG_DEFAULTS,
    evaluation_config=None,
):
    # Update local tuning_config (in current recursion stack)
    local_config_path = curr_dir.joinpath(constants.CONFIG_FILENAME)
    if os.path.exists(local_config_path):
        tuning_config = open_config_with_defaults(local_config_path)

    # Update local template (in current recursion stack)
    local_template_path = curr_dir.joinpath(constants.TEMPLATE_FILENAME)
    local_template_exists = os.path.exists(local_template_path)
    if local_template_exists:
        template = Template(
            local_template_path,
            tuning_config,
        )
    # Look for subdirectories for processing
    subdirs = [d for d in curr_dir.iterdir() if d.is_dir()]

    output_dir = Path(args["output_dir"], curr_dir.relative_to(root_dir))
    paths = Paths(output_dir)

    # look for images in current dir to process
    exts = ("*.[pP][nN][gG]", "*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]")
    omr_files = sorted([f for ext in exts for f in curr_dir.glob(ext)])

    # Exclude images (take union over all pre_processors)
    excluded_files = []
    if template:
        for pp in template.pre_processors:
            excluded_files.extend(Path(p) for p in pp.exclude_files())

    local_evaluation_path = curr_dir.joinpath(constants.EVALUATION_FILENAME)
    if not args["setLayout"] and os.path.exists(local_evaluation_path):
        if not local_template_exists:
            logger.warning(
                f"Found an evaluation file without a parent template file: {local_evaluation_path}"
            )
        evaluation_config = EvaluationConfig(
            curr_dir,
            local_evaluation_path,
            template,
            tuning_config,
        )

        excluded_files.extend(
            Path(exclude_file) for exclude_file in evaluation_config.get_exclude_files()
        )

    omr_files = [f for f in omr_files if f not in excluded_files]

    if omr_files:
        if not template:
            logger.error(
                f"Found images, but no template in the directory tree \
                of '{curr_dir}'. \nPlace {constants.TEMPLATE_FILENAME} in the \
                appropriate directory."
            )
            raise Exception(
                f"No template file found in the directory tree of {curr_dir}"
            )

        setup_dirs_for_paths(paths)
        outputs_namespace = setup_outputs_for_template(paths, template)

        print_config_summary(
            curr_dir,
            omr_files,
            template,
            tuning_config,
            local_config_path,
            evaluation_config,
            args,
        )
        if args["setLayout"]:
            show_template_layouts(omr_files, template, tuning_config)
        else:
            process_files(
                omr_files,
                template,
                tuning_config,
                evaluation_config,
                outputs_namespace,
            )

    elif not subdirs:
        # Each subdirectory should have images or should be non-leaf
        logger.info(
            f"No valid images or sub-folders found in {curr_dir}.\
            Empty directories not allowed."
        )

    # recursively process sub-folders
    for d in subdirs:
        process_dir(
            root_dir,
            d,
            args,
            template,
            tuning_config,
            evaluation_config,
        )


def show_template_layouts(omr_files, template, tuning_config):
    # Create debug directory if debugging is enabled
    debug_dir = Path("debug")
    if tuning_config.outputs.show_image_level >= 1:
        debug_dir.mkdir(exist_ok=True)

    for file_path in omr_files:
        file_name = file_path.name
        file_basename = Path(file_name).stem
        file_path = str(file_path)

        # Save original image
        in_omr = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if tuning_config.outputs.show_image_level >= 1:
            debug_image_path = debug_dir.joinpath(f"{file_basename}_01_original.jpg")
            ImageUtils.save_img(str(debug_image_path), in_omr)

        # Save image after preprocessors
        in_omr = template.image_instance_ops.apply_preprocessors(
            file_path, in_omr, template
        )
        if tuning_config.outputs.show_image_level >= 1:
            debug_image_path = debug_dir.joinpath(f"{file_basename}_02_preprocessed.jpg")
            ImageUtils.save_img(str(debug_image_path), in_omr)

        # Save template layout - without resizing
        template_layout = template.image_instance_ops.draw_template_layout(
            in_omr, template, shifted=False, border=2, preserve_size=False
        )
        if tuning_config.outputs.show_image_level >= 1:
            debug_image_path = debug_dir.joinpath(f"{file_basename}_03_template_layout.jpg")
            ImageUtils.save_img(str(debug_image_path), template_layout)
        InteractionUtils.show(
            f"Template Layout: {file_name}", template_layout, 1, 1, config=tuning_config
        )


def process_files(
    omr_files,
    template,
    tuning_config,
    evaluation_config,
    outputs_namespace,
):
    start_time = int(time())
    files_counter = 0
    STATS.files_not_moved = 0

    # Create debug directory if debugging is enabled
    debug_dir = outputs_namespace.paths.output_dir.joinpath("debug")
    if tuning_config.outputs.show_image_level >= 1:
        debug_dir.mkdir(exist_ok=True)

    for file_path in omr_files:
        files_counter += 1
        file_name = file_path.name
        file_basename = Path(file_name).stem

        in_omr = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

        if tuning_config.outputs.show_image_level >= 1:
            # Save original image
            debug_image_path = debug_dir.joinpath(f"{file_basename}_01_original.jpg")
            ImageUtils.save_img(str(debug_image_path), in_omr)

        logger.info("")
        logger.info(
            f"({files_counter}) Opening image: \t'{file_path}'\tResolution: {in_omr.shape}"
        )

        template.image_instance_ops.reset_all_save_img()
        template.image_instance_ops.append_save_img(1, in_omr)

        # Apply preprocessors and track steps
        processed_omr = template.image_instance_ops.apply_preprocessors(
            file_path, in_omr, template
        )

        # Save debug images if enabled
        if tuning_config.outputs.show_image_level >= 1:
            for i, preprocessor in enumerate(template.pre_processors):
                if processed_omr is not None:
                    debug_image_path = debug_dir.joinpath(f"{file_basename}_02_{i+1}_after_{preprocessor.__class__.__name__}.jpg")
                    ImageUtils.save_img(str(debug_image_path), processed_omr)

        in_omr = processed_omr

        if in_omr is None:
            # Error OMR case
            new_file_path = outputs_namespace.paths.errors_dir.joinpath(file_name)
            outputs_namespace.OUTPUT_SET.append(
                [file_name] + outputs_namespace.empty_resp
            )
            if check_and_move(
                constants.ERROR_CODES.NO_MARKER_ERR, file_path, new_file_path
            ):
                err_line = [
                    file_name,
                    file_path,
                    new_file_path,
                    "NA",
                ] + outputs_namespace.empty_resp
                pd.DataFrame(err_line, dtype=str).T.to_csv(
                    outputs_namespace.files_obj["Errors"],
                    mode="a",
                    quoting=QUOTE_NONNUMERIC,
                    header=False,
                    index=False,
                )
            continue

        # Store original auto_align setting and disable it temporarily
        orig_auto_align = tuning_config.alignment_params.auto_align
        tuning_config.alignment_params.auto_align = False

        # Create template layout in setLayout style
        template_layout = template.image_instance_ops.draw_template_layout(
            in_omr, template, shifted=False, border=2, preserve_size=False
        )
        if tuning_config.outputs.show_image_level >= 1:
            debug_image_path = debug_dir.joinpath(f"{file_basename}_02_template_layout.jpg")
            ImageUtils.save_img(str(debug_image_path), template_layout)
            InteractionUtils.show(
                f"Template Layout: {file_name}", template_layout, 1, 1, config=tuning_config
            )

        # uniquify
        file_id = str(file_name)
        save_dir = outputs_namespace.paths.save_marked_dir
        (
            response_dict,
            final_marked,
            multi_marked,
            _,
        ) = template.image_instance_ops.read_omr_response(
            template, image=in_omr, name=file_id, save_dir=save_dir, preserve_size=False
        )

        # Save final marked OMR image
        if tuning_config.outputs.show_image_level >= 1:
            debug_image_path = debug_dir.joinpath(f"{file_basename}_03_final_marked.jpg")
            ImageUtils.save_img(str(debug_image_path), final_marked)

        # --- Custom Summation Logic for Part D ---
        part_d_raw_list = []
        part_d_sum = 0
        # Use the number of items defined by the template
        part_d_block = next((fb for fb in template.field_blocks if fb.name == "Part_D_Block"), None)
        num_part_d_items = 0
        if part_d_block:
            num_part_d_items = len(part_d_block.parsed_field_labels)
        else:
            logger.warning(f"Part_D_Block not found in template for {file_name}")

        for i in range(1, num_part_d_items + 1):
            label = f'part_d{i}'  # Key as defined in template.json fieldLabels
            # Get the digit from the response_dict
            digit_str = response_dict.get(label, '')
            part_d_raw_list.append(str(digit_str))
            try:
                digit_val = int(digit_str)
                part_d_sum += digit_val
            except (ValueError, TypeError):
                pass  # Ignore non-integer values for sum
        part_d_raw = "".join(part_d_raw_list)

        # --- Custom Summation Logic for Part E (Marking Block) ---
        marking_raw_list = []
        marking_sum = 0
        marking_block = next((fb for fb in template.field_blocks if fb.name == "Marking_Block"), None)
        num_marking_items = 0
        if marking_block:
            num_marking_items = len(marking_block.parsed_field_labels)
        else:
            logger.warning(f"Marking_Block not found in template for {file_name}")

        for i in range(1, num_marking_items + 1):  # Should be 3 items (marking1..3)
            label = f'marking{i}'
            digit_str = response_dict.get(label, '')
            marking_raw_list.append(str(digit_str))
            try:
                digit_val = int(digit_str)
                marking_sum += digit_val
            except (ValueError, TypeError):
                pass
        marking_raw = "".join(marking_raw_list)

        # Get APAAR ID and Exam Code
        apaar_id = response_dict.get('Roll', 'MISSING_ID')
        exam_code = response_dict.get('Int_Block_Q1', 'MISSING_CODE')

        # TODO: move inner try catch here
        # concatenate roll nos, set unmarked responses, etc
        omr_response = get_concatenated_response(response_dict, template)

        if (
            evaluation_config is None
            or not evaluation_config.get_should_explain_scoring()
        ):
            logger.info(f"Read Response: \n{omr_response}")

        # Evaluate answers and calculate score
        score = 0
        for question_id, answer_data in answer_key.items():
            correct_answer = answer_data["correct_answer"]
            points = answer_data["points"]
            user_answer = response_dict.get(question_id, "")
            if user_answer == correct_answer:
                score += points

        if evaluation_config is not None:
            score = evaluate_concatenated_response(
                omr_response, evaluation_config, file_path, outputs_namespace.paths.evaluation_dir
            )
            logger.info(
                f"(/{files_counter}) Graded with score: {round(score, 2)}\t for file: '{file_id}'"
            )
        else:
            logger.info(f"(/{files_counter}) Processed file: '{file_id}'")

        if tuning_config.outputs.show_image_level >= 2:
            InteractionUtils.show(
                f"Final Marked Bubbles : '{file_id}'",
                ImageUtils.resize_util_h(
                    final_marked, int(tuning_config.dimensions.display_height * 1.3)
                ),
                1,
                1,
                config=tuning_config,
            )

        resp_array = []
        for k in template.output_columns:
            resp_array.append(omr_response[k])

        outputs_namespace.OUTPUT_SET.append([file_name] + resp_array)

        if multi_marked == 0 or not tuning_config.outputs.filter_out_multimarked_files:
            STATS.files_not_moved += 1
            new_file_path = save_dir.joinpath(file_id)
            # Ensure the number of columns in results_line matches result_csv_header
            # Adjust the results_line to include only the required columns
            results_line = [
                file_name,               # ImagePath (or filename)
                str(file_path),          # FilePath (original)
                str(new_file_path),      # NewFilePath
                0,                       # Initialize Total Score
                apaar_id,                # APAAR ID for reference
                marking_raw,             # The raw captured markings
                part_d_raw               # The raw captured digits
            ]

            # Add data for each question: marked answer, correct answer, and marks
            score = 0
            for question_id, answer_data in answer_key.items():
                correct_answer = answer_data["correct_answer"]
                points = answer_data["points"]
                user_answer = response_dict.get(question_id, "")
                marks = points if user_answer == correct_answer else 0
                score += marks

                # Append marked answer, correct answer, and marks for the question
                results_line.extend([user_answer, correct_answer, marks])

            # Update the total score in the results_line
            results_line[3] = score  # Update the score column

            # Write the updated results_line to the Results CSV file
            pd.DataFrame([results_line], columns=result_csv_header).to_csv(
                outputs_namespace.files_obj["Results"],
                mode="a",
                quoting=QUOTE_NONNUMERIC,
                header=False,
                index=False,
            )
        else:
            # multi_marked file
            logger.info(f"[{files_counter}] Found multi-marked file: '{file_id}'")
            new_file_path = outputs_namespace.paths.multi_marked_dir.joinpath(file_name)
            if check_and_move(
                constants.ERROR_CODES.MULTI_BUBBLE_WARN, file_path, new_file_path
            ):
                mm_line = [file_name, file_path, new_file_path, "NA"] + resp_array
                pd.DataFrame(mm_line, dtype=str).T.to_csv(
                    outputs_namespace.files_obj["MultiMarked"],
                    mode="a",
                    quoting=QUOTE_NONNUMERIC,
                    header=False,
                    index=False,
                )
            # else:
            #     TODO:  Add appropriate record handling here
            #     pass

        # Restore original auto_align setting
        tuning_config.alignment_params.auto_align = orig_auto_align

    print_stats(start_time, files_counter, tuning_config)


def check_and_move(error_code, file_path, filepath2):
    # TODO: fix file movement into error/multimarked/invalid etc again
    STATS.files_not_moved += 1
    return True


def print_stats(start_time, files_counter, tuning_config):
    time_checking = max(1, round(time() - start_time, 2))
    log = logger.info
    log("")
    log(f"{'Total file(s) moved': <27}: {STATS.files_moved}")
    log(f"{'Total file(s) not moved': <27}: {STATS.files_not_moved}")
    log("--------------------------------")
    log(
        f"{'Total file(s) processed': <27}: {files_counter} ({'Sum Tallied!' if files_counter == (STATS.files_moved + STATS.files_not_moved) else 'Not Tallying!'})"
    )

    if tuning_config.outputs.show_image_level <= 0:
        log(
            f"\nFinished Checking {files_counter} file(s) in {round(time_checking, 1)} seconds i.e. ~{round(time_checking / 60, 1)} minute(s)."
        )
        log(
            f"{'OMR Processing Rate': <27}: \t ~ {round(time_checking / files_counter, 2)} seconds/OMR"
        )
        log(
            f"{'OMR Processing Speed': <27}: \t ~ {round((files_counter * 60) / time_checking, 2)} OMRs/minute"
        )
    else:
        log(f"\n{'Total script time': <27}: {time_checking} seconds")

    if tuning_config.outputs.show_image_level <= 1:
        log(
            "\nTip: To see some awesome visuals, open config.json and increase 'show_image_level'"
        )


# Ensure the answer_key is loaded before it is used in the process_files function
answer_key_path = "samples/answer-key/using-csv/answer_key.csv"
answer_key = {}
try:
    with open(answer_key_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            answer_key[row["question_id"]] = {
                "correct_answer": row["correct_answer"],
                "points": int(row["points"]),
            }
except FileNotFoundError:
    logger.error(f"Answer key file not found at: {answer_key_path}")
    exit(1)
except Exception as e:
    logger.error(f"Error loading answer key: {e}")
    exit(1)
