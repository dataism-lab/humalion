"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """Humalion."""


if __name__ == "__main__":
    main(prog_name="humalion")  # pragma: no cover
