#!/usr/bin/env python
"""
An example of a step using MLflow and Weights & Biases]: Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact

    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()
    filename = args.output_artifact

    logger.info("Loading artifact to dataframe")
    df = pd.read_csv(artifact_path)

    logger.info("Filter Dataframe before saving it in wandb")
    df['last_review'] = pd.to_datetime(df['last_review'])
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    logger.info("save the file")
    df.to_csv(filename, index=False)

    logger.info("Creating artifact")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    os.remove(filename)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help='input artifact to read from and clean',
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help='name of the cleaned output artifact',
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help='type of cleaned output artifact',
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help='short description what has been done ',
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help='minimum price to filter entries with a too low price',
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help='minimum price to filter entries with a outlying high price',
        required=True
    )


    args = parser.parse_args()

    go(args)
