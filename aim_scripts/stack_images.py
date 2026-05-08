from PIL import Image

# Load the two PNG images
top_image = Image.open("full_diagram.png").convert("RGBA")
bottom_image = Image.open("ev_png/evolutionary_search_ua_horizontal.png").convert("RGBA")

# Resize images if needed (optional)
# Example: make both images the same width
max_width = max(top_image.width, bottom_image.width)

def resize_to_width(img, width):
    if img.width == width:
        return img
    ratio = width / img.width
    new_height = int(img.height * ratio)
    return img.resize((width, new_height), Image.LANCZOS)

top_image = resize_to_width(top_image, max_width)
bottom_image = resize_to_width(bottom_image, max_width)

# Create a new blank image tall enough to hold both
combined_height = top_image.height + bottom_image.height

combined_image = Image.new(
    "RGBA",
    (max_width, combined_height),
    (255, 255, 255, 0)  # transparent background
)

# Paste images
combined_image.paste(top_image, (0, 0))
combined_image.paste(bottom_image, (0, top_image.height))

# Save result
combined_image.save("stacked.png")

print("Saved stacked image as stacked.png")