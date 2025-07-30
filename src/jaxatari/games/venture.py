
























"""sprites please leave it at the end"""

def _load_sprites(self) -> dict[str, Any]:
    """Loads all necessary sprites from .npy files."""
sprites: Dict[str, Any] = {}       
       
       
import jaxatari.rendering.atraJaxis as aj
from jaxatari.renderers import AtraJaxisRenderer
        
class VentureRenderer(AtraJaxisRenderer):
    # Type hint for sprites dictionary
    sprites: Dict[str, Any]       
    
    def __init__(self):
        """
        Initializes the renderer by loading sprites, including level backgrounds.

        Args:
            sprite_path: Path to the directory containing sprite .npy files.
        """
        self.sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/venture"
        self.sprites = self._load_sprites()
        # Store background sprites directly for use in render function
        
        """
        ///////////////////////////////////////
        self.background_0 = self.sprites.get('background_0')
        self.background_1 = self.sprites.get('background_1')
        self.background_2 = self.sprites.get('background_2')
        """

    def _load_sprites(self) -> dict[str, Any]:
        """Loads all necessary sprites from .npy files."""
        sprites: Dict[str, Any] = {}
        
        # Helper function to load a single sprite frame
        def _load_sprite_frame(name: str) -> Optional[chex.Array]:
            path = os.path.join(self.sprite_path, f'{name}.npy')
            frame = aj.loadFrame(path)
            if isinstance(frame, jnp.ndarray) and frame.ndim >= 2:
                return frame.astype(jnp.uint8)
            
        
        # --- Load Sprites ---
        # Backgrounds + Dynamic elements + UI elements
        # background_0 is black background
        sprite_names = [
            ‘venture_background0', ‘venture_background1', 
            ‘venture_purple_monster', ‘venture_green_monster', 'venture_dot',
        ]
        for name in sprite_names:
            loaded_sprite = _load_sprite_frame(name)
            if loaded_sprite is not None:
                 sprites[name] = loaded_sprite    
            
        
        # pad the kangaroo and monkey sprites since they have to be used interchangeably (and jax enforces same sizes)???



         # expand all sprites similar to the Pong/Seaquest loading
        for key in sprites.keys():
            if isinstance(sprites[key], (list, tuple)):
                sprites[key] = [jnp.expand_dims(sprite, axis=0) for sprite in sprites[key]]
            else:
                sprites[key] = jnp.expand_dims(sprites[key], axis=0)

        return sprites
            
