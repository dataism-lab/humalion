"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Pygmalion."""


if __name__ == "__main__":
    main(prog_name="pygmalion")  # pragma: no cover
