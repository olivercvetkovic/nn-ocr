import pygame
import numpy as np


def draw_number(file_path: str):
    pygame.init()

    # High resolution drawing canvas (10x MNIST size)
    canvas_size = 128  # e.g., 280x280 pixels
    screen_size = canvas_size
    screen = pygame.display.set_mode((screen_size, screen_size + 50))
    pygame.display.set_caption("Draw Number")

    # Create a high-res drawing surface with black background
    canvas = pygame.Surface((canvas_size, canvas_size))
    canvas.fill((0, 0, 0))

    # Button properties
    button_color = (50, 205, 50)
    button_rect = pygame.Rect(screen_size//2 - 40, screen_size + 10, 80, 30)

    running = True
    drawing = False

    clock = pygame.time.Clock()

    while running:
        clock.tick(120)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Handle mouse events
            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    # Downscale to 28x28 with smooth scaling (anti-aliasing)
                    scaled_image = pygame.transform.smoothscale(canvas, (28, 28))
                    pygame.image.save(scaled_image, file_path)
                    print(f"Image saved as '{file_path}'")
                    pygame.display.quit()  # Close the display
                    pygame.quit()          # Uninitialize all pygame modules
                    return
                else:
                    drawing = True

            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False

            if event.type == pygame.MOUSEMOTION and drawing:
                x, y = event.pos
                if y < screen_size:  # Only draw in the canvas area
                    # Draw a circle for a smoother (anti-aliased) stroke
                    brush_radius = 4.57  # Adjust for desired stroke thickness
                    pygame.draw.circle(canvas, (255, 255, 255), (x, y), brush_radius)

        # Clear screen and blit the current canvas
        screen.fill((0, 0, 0))
        screen.blit(canvas, (0, 0))

        # Draw grid border (optional) and button
        pygame.draw.rect(screen, button_color, button_rect)
        font = pygame.font.Font(None, 24)
        text = font.render("Save", True, (255, 255, 255))
        text_rect = text.get_rect(center=button_rect.center)
        screen.blit(text, text_rect)

        pygame.display.flip()

    pygame.quit()


def vectorise_image(image_file: str) -> np.ndarray:
    """Convert a 28x28 PNG image of a drawn digit into an MNIST-compatible vector.
    
    Args:
        image_file: Path to the PNG file created by draw_number()
    
    Returns:
        Numpy array of shape (784, 1) with normalized pixel values (0.0-1.0),
        where the background is white (1.0) and the digit is black (0.0).
    """
    # Load the image using Pygame
    image = pygame.image.load(image_file)
    
    # Store pixel values
    pixels = []
    
    # Extract and normalize pixels with inversion
    for y in range(28):
        for x in range(28):
            # Get RGB values (image is black background with white drawing)
            r, g, b, *_ = image.get_at((x, y))
            
            # Normalize and invert:
            # For a drawn image with white digit on black background:
            # - White (r=255) becomes 0.0 (digit in MNIST)
            # - Black (r=0) becomes 1.0 (background in MNIST)
            normalized = (r * (0.99 / 255.0))
            pixels.append(normalized)
    
    # Convert to numpy array and reshape to match MNIST format
    return np.array(pixels).reshape((-1, 1)).astype(np.float64)
    