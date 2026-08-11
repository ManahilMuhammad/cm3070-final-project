import instrumentation as inst
from io import BytesIO
import hashlib
import imagehash


@inst.timed("fusion of extracted text")
def fuse(transcript, slides, notes, figure):
    """
    combines all pieces of extracted text together
    """

    parts = []
    components = [
        {'name': "TRANSCRIPT", 'content': transcript}, 
        {'name': "SLIDE", 'content': slides}, 
        {'name': "NOTES", 'content': notes}, 
        {'name': "FIGURE", 'content': figure}
    ]

    for component in components:
        content = component['content'].strip()
        if content and content.lower() != "none":
            parts.append(f"[{component['name']}]\n{content}")

    fusion = '\n\n'.join(parts)

    return fusion


def image_hash(img):
    """
    returns unique hash of image (used to detect duplicates)
    """
    buffer = BytesIO()
    img.convert("RGB").save(buffer, format="PNG")

    return hashlib.md5(buffer.getvalue()).hexdigest


def deduplicate_images(images):
    """
    returns images list after removing duplicates
    """
    seen = set()
    unique = []

    for img in images:
        h = image_hash(img)

        if h not in seen:
            seen.add(h)
            unique.append(img)

    return unique


def resize_image(img, max_size=1024):
    img = img.convert("RGB")

    width, height = img.size

    scale = min(1.0, max_size / max(width, height))

    if scale < 1.0:
        new_size = (
            int(width * scale),
            int(height * scale)
        )

        img = img.resize(new_size)

    return img


def image_phash(images, max_dist=5):
    """
    removes visually similar/duplicate images using perceptual hashing
    """
    unique = []
    hashes = []

    for img in images:
        # convert to RGB for uniformity
        img_rgb = img.convert("RGB")

        # calculate perceptual hash
        h = imagehash.phash(img_rgb)

        # whether image is visually similar
        is_dupe = False

        for hash in hashes:
            dist = h - hash

            if dist <= max_dist:
                is_dupe = True # duplicate detected
                break

        if not is_dupe:
            hashes.append(h)
            unique.append(img)

    return unique


def prepare_images(images, max_num=10):
    """
    prepare images list by removing tiny, duplicate,
    decorative images, sorting by size, and returning top few
    """

    # remove tiny images (logo, etc)
    images = [
        img for img in images
        if img.width >= 150
        and img.height >= 100
        and img.width * img.height >= 30_000
        and max(img.width/img.height, img.height/img.width) <= 8
    ]

    # remove exact duplicates
    images = deduplicate_images(images)

    # remove visual duplicates
    images = image_phash(images)

    # resize images to make them smaller for llava
    images = [
        resize_image(img) for img in images
    ]

    # rank images by size
    images.sort(
        key=lambda img: img.width * img.height,
        reverse=True
    )

    # limit number of images
    return images[:max_num]