"""CLI entry point."""

from __future__ import annotations

import click


@click.group()
@click.version_option()
def main() -> None:
    """Ununennium: Satellite imagery ML toolkit."""
    pass


@main.command()
@click.option("--config", "-c", required=True, help="Path to training config file")
@click.option("--output", "-o", default="outputs", help="Output directory")
@click.option("--resume", type=click.Path(exists=True), help="Resume from checkpoint")
def train(config: str, output: str, resume: str | None) -> None:
    """Train a model from configuration."""
    click.echo(f"Training with config: {config}")
    click.echo(f"Output directory: {output}")
    if resume:
        click.echo(f"Resuming from: {resume}")
    # TODO: Implement training


@main.command()
@click.option("--model", "-m", required=True, help="Path to model checkpoint")
@click.option("--data", "-d", required=True, help="Path to evaluation data")
@click.option("--output", "-o", default="results", help="Output directory")
def evaluate(model: str, data: str, output: str) -> None:
    """Evaluate a trained model."""
    click.echo(f"Evaluating model: {model} on data: {data}")
    click.echo(f"Results will be saved to: {output}")
    click.echo(f"Data: {data}")
    # TODO: Implement evaluation


@main.command()
@click.option("--model", "-m", required=True, help="Path to model checkpoint")
@click.option("--format", "-f", type=click.Choice(["onnx", "torchscript"]), default="onnx")
@click.option("--output", "-o", required=True, help="Output path")
def export(model: str, format: str, output: str) -> None:
    """Export model to deployment format."""
    click.echo(f"Exporting {model} to {format}: {output}")
    # TODO: Implement export


if __name__ == "__main__":
    main()
