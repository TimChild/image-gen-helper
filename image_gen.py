import os
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
from datetime import datetime
from typing import Literal
from openai import AsyncOpenAI
from dotenv import load_dotenv
import base64
import re
import logging
from openai.types import ImagesResponse
from rich.logging import RichHandler
from IPython.display import Image, display, Markdown
from openai.types.image import Image as OpenaiImage
from IPython.core.getipython import get_ipython

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)


SAVE_DIR = "images"

# Load environment variables from a .env file if present
load_dotenv()

# Check for OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError(
        "OpenAI API key not found. Please set it as an environment variable or in a .env file."
    )

client = AsyncOpenAI(api_key=api_key)


def sanitize_filename(filename: str) -> str:
    """Sanitize the filename to remove or replace invalid characters."""
    return re.sub(r'[<>:"/\\|?*]', "_", filename)


@dataclass
class ImageInfo:
    image: OpenaiImage = field(repr=False)
    path: Path


async def generate_images(
    prompt: str | list[str],
    n: int = 1,
    shape: Literal["square", "portrait", "landscape"] = "square",
    style: Literal["vivid", "natural"] = "vivid",
    quality: Literal["standard", "hd"] = "standard",
) -> list[ImageInfo]:
    """
    Generate images using DALL-E 3 with the given prompt and save them.

    Args:
        prompt: Description of the image to generate.
        n: Number of images to generate (default is 1).
        shape: Desired shape of the image.
        style: Desired style of the image.
    Returns:
        A list of paths to the generated images.
    """
    os.makedirs("images", exist_ok=True)

    prompt_list: list[str] = [prompt] if isinstance(prompt, str) else prompt

    tasks = []
    async with asyncio.TaskGroup() as tg:
        for prompt in prompt_list:
            for i in range(n):
                tasks.append(
                    tg.create_task(_generate_image(prompt, shape, style, quality, i=i))
                )

    infos: list[ImageInfo] = await asyncio.gather(*tasks)
    if get_ipython():
        for info in infos:
            _display_image(info)

    assert all(isinstance(info, ImageInfo) for info in infos)
    return infos


def _display_image(info: ImageInfo) -> None:
    display(
        Markdown(f"**Path:** {info.path}\n\n**Prompt:**\n\n{info.image.revised_prompt}")
    )
    display(Image(info.path))
    display(Markdown("---"))


async def _generate_image(
    prompt: str,
    shape: Literal["square", "portrait", "landscape"] = "square",
    style: Literal["vivid", "natural"] = "vivid",
    quality: Literal["standard", "hd"] = "standard",
    i: int = 0,
) -> ImageInfo:
    size_mapping: dict[str, Literal["1024x1024", "1024x1792", "1792x1024"]] = {
        "square": "1024x1024",
        "portrait": "1024x1792",
        "landscape": "1792x1024",
    }
    size = size_mapping.get(shape, "1024x1024")

    try:
        response: ImagesResponse = await client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,  # Only allows one...
            quality=quality,
            size=size,
            style=style,
            response_format="b64_json",
        )
        image_data = response.data[0]

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_prompt = sanitize_filename(prompt[:10])
        filename = f"images/{timestamp}-n{i + 1}-{safe_prompt}"

        assert image_data.b64_json is not None
        image = base64.b64decode(image_data.b64_json)

        image_save_path = Path(filename + ".png")
        info = ImageInfo(image=image_data, path=image_save_path)

        with open(image_save_path, "wb") as f:
            f.write(image)

        with open(filename + ".txt", "w") as f:
            f.write(
                f"Initial prompt: {prompt}\nRevised prompt:\n{image_data.revised_prompt}"
            )

        logging.info(f"Image saved as {filename}")

        return info
    except Exception as e:
        # logging.exception(f"An error occurred: {e}")
        raise e


def display_all_images_for(prompt: str) -> list[Path]:
    """Load and display all images that included the given prompt text.

    Args:
        - prompt: Text that was included in the prompt for the images.
    """
    # First look through all the `.txt` files in the images directory, check which of those contain the prompt text
    all_paths: list[Path] = []
    for filename in Path(SAVE_DIR).glob("*.txt"):
        with open(filename, "r") as f:
            if prompt in f.read():
                # If the prompt text is found, display the image with the same filename (minus the `.txt` and add `.png`)
                image_filename = filename.with_suffix(".png")
                display(Image(image_filename, alt=image_filename.name))
                logging.info(f"Displayed image: {image_filename}")
                all_paths.append(image_filename)
    return all_paths
