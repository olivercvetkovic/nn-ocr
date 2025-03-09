import pygame


def draw_number():
    # Initialize Pygame
    pygame.init()
    
    # Grid dimensions and scaling
    grid_size = 28
    scale_factor = 20
    window_size = grid_size * scale_factor
    screen = pygame.display.set_mode((window_size, window_size + 50))
    pygame.display.set_caption("Draw Number")
    
    # Create drawing surface
    drawing_surface = pygame.Surface((grid_size, grid_size))
    drawing_surface.fill((0, 0, 0))
    
    # Button properties
    button_color = (50, 205, 50)
    button_rect = pygame.Rect(window_size//2 - 40, window_size + 10, 80, 30)
    
    # Main loop variables
    running = True
    drawing = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Handle mouse events
            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    # Save the image when button is clicked
                    pygame.image.save(drawing_surface, "drawn_number.png")
                    print("Image saved as 'drawn_number.png'")
                else:
                    drawing = True
                    
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                
            if event.type == pygame.MOUSEMOTION and drawing:
                # Convert screen coordinates to grid coordinates
                x, y = event.pos
                if y < window_size:  # Only draw in the grid area
                    grid_x = x // scale_factor
                    grid_y = y // scale_factor
                    if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                        # Draw on the 28x28 surface
                        drawing_surface.set_at((grid_x, grid_y), (255, 255, 255))
        
        # Clear screen
        screen.fill((0, 0, 0))
        
        # Draw the scaled-up grid
        for y in range(grid_size):
            for x in range(grid_size):
                pixel_color = drawing_surface.get_at((x, y))
                rect = pygame.Rect(x*scale_factor, y*scale_factor, 
                                 scale_factor, scale_factor)
                pygame.draw.rect(screen, pixel_color, rect)
        
        # Draw grid lines
        for x in range(0, window_size, scale_factor):
            pygame.draw.line(screen, (40, 40, 40), (x, 0), (x, window_size))
        for y in range(0, window_size, scale_factor):
            pygame.draw.line(screen, (40, 40, 40), (0, y), (window_size, y))
        
        # Draw save button
        pygame.draw.rect(screen, button_color, button_rect)
        font = pygame.font.Font(None, 24)
        text = font.render("Save", True, (255, 255, 255))
        text_rect = text.get_rect(center=button_rect.center)
        screen.blit(text, text_rect)
        
        pygame.display.flip()

    pygame.quit()


# Run the function
draw_number()