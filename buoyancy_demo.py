# === Buoyancy (Z-up) — Isaac Sim 5.0 (Script Editor) - VERSÃO FINAL E ROBUSTA ===
# - Funciona na cena atual, sem salvar o Stage.
# - Água em Z=0, gravidade -Z.
# - Usa uma subscrição direta ao evento de física para máxima robustez.
# - Contém depuração explícita para diagnosticar a leitura da posição do cubo.

import omni.usd
import omni.timeline
import omni.physx
from pxr import UsdGeom, UsdPhysics, Gf, Sdf
import carb

class BuoyancyInEditor:
    """
    Encapsula a lógica de flutuabilidade para funcionar dentro do Script Editor do Isaac Sim.
    Gere o estado e a subscrição ao evento de física de forma robusta.
    """
    def __init__(self, cube_path="/World/BuoyantCube"):
        # --- Parâmetros de Simulação (Ajustados para garantir a flutuação) ---
        self.CUBE_PATH = Sdf.Path(cube_path)
        self.CUBE_SIZE = 1.0      # m (Volume = 1m³)
        self.WATER_Z   = 0.0      # Nível da água no eixo Z
        self.RHO       = 1000.0   # Densidade da água (kg/m^3)
        self.G         = 9.81     # Módulo da gravidade (m/s^2)
        
        # Para um cubo de 1m³ flutuar em água com densidade 1000, a massa DEVE ser < 1000 kg.
        self.MASS      = 500.0    # Massa do cubo em kg.
        
        # Coeficientes de amortecimento para estabilidade
        self.DRAG_Z    = 80.0     # Arrasto linear vertical (N·s/m)
        self.DRAG_XY   = 50.0     # Arrasto linear lateral (N·s/m)
        self.ANGULAR_DRAG = 50.0  # Arrasto angular para estabilizar a rotação

        # --- Variáveis de Estado ---
        self.cube_prim = None
        self.physx_interface = omni.physx.get_physx_interface()
        self.physx_subscription = None
        
        # --- Obter interfaces principais ---
        self.stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()

    def setup_scene(self):
        """Garante que a cena de física e o cubo existem e estão corretamente configurados."""
        # 1. Garantir que a cena de física existe
        scene_prim = self.stage.GetPrimAtPath("/World/PhysicsScene")
        if not scene_prim.IsValid():
            UsdPhysics.Scene.Define(self.stage, "/World/PhysicsScene")
            scene = UsdPhysics.Scene(self.stage.GetPrimAtPath("/World/PhysicsScene"))
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
            scene.CreateGravityMagnitudeAttr().Set(self.G)
            carb.log_info("Cena de física criada em /World/PhysicsScene.")

        # 2. Criar o cubo se ele não existir
        self.cube_prim = self.stage.GetPrimAtPath(self.CUBE_PATH)
        if not self.cube_prim.IsValid():
            carb.log_info(f"Cubo não encontrado. A criar um novo em {self.CUBE_PATH}.")
            cube = UsdGeom.Cube.Define(self.stage, self.CUBE_PATH)
            cube.CreateSizeAttr(self.CUBE_SIZE)
            
            xform = UsdGeom.Xformable(cube.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3f(0, 0, 2.0)) # Posição inicial acima da água

            # Aplicar APIs de física
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
            mass_api = UsdPhysics.MassAPI.Apply(cube.GetPrim())
            mass_api.CreateMassAttr().Set(self.MASS)
            self.cube_prim = cube.GetPrim()
        else:
            carb.log_info(f"A usar o cubo existente em {self.CUBE_PATH}.")

    def _on_physics_step(self, dt: float):
        """
        Callback executado a cada passo de física. Calcula e aplica as forças.
        """
        if not self.timeline.is_playing() or not self.cube_prim or not self.cube_prim.IsValid():
            return

        body_path = self.cube_prim.GetPath().pathString
        try:
            body_transform = self.physx_interface.get_rigidbody_transformation(body_path)
            position = Gf.Vec3d(body_transform["position"])
            
            # --- PONTO DE DEPURAÇÃO CRÍTICO ---
            # Imprime a posição Z que o script está a ler em todos os frames.
            carb.log_warn(f"DEBUG: Posição Z lida pelo script: {position[1]:.4f}")

            body_velocity = self.physx_interface.get_rigidbody_velocity(body_path)
            linear_velocity = Gf.Vec3f(body_velocity["linear_velocity"])
            angular_velocity = Gf.Vec3f(body_velocity["angular_velocity"])
        except Exception as e:
            # Se houver um erro ao ler a física, não fazemos nada.
            return

        # 2. Calcular a fração submersa
        half_size = 0.5 * self.CUBE_SIZE
        z_bottom = position[1] - half_size
        
        submerged_depth = max(0.0, min(self.CUBE_SIZE, self.WATER_Z - z_bottom))
        
        if submerged_depth <= 1e-6:
            return # Fora de água, não aplicar forças.

        # Se o código chegar aqui, significa que o cubo está submerso.
        carb.log_info(">>> CUBO SUBMERSO DETETADO! A CALCULAR FORÇAS...")

        # 3. Calcular Força de Empuxo
        submerged_volume = submerged_depth * (self.CUBE_SIZE * self.CUBE_SIZE)
        buoyancy_force_magnitude = self.RHO * self.G * submerged_volume
        buoyancy_force = carb.Float3(0.0, 0.0, buoyancy_force_magnitude)

        # O empuxo atua no centro do volume submerso para estabilizar
        force_pos_z = z_bottom + submerged_depth / 2.0
        force_position = carb.Float3(position, position[2], force_pos_z)

        # 4. Calcular Forças de Arrasto (Amortecimento)
        drag_force = carb.Float3(
            -self.DRAG_XY * linear_velocity,
            -self.DRAG_XY * linear_velocity[2],
            -self.DRAG_Z * linear_velocity[1]
        )
        drag_torque = carb.Float3(
            -self.ANGULAR_DRAG * angular_velocity,
            -self.ANGULAR_DRAG * angular_velocity[2],
            -self.ANGULAR_DRAG * angular_velocity[1]
        )

        # 5. Aplicar as forças
        self.physx_interface.apply_force_at_pos(body_path, buoyancy_force, force_position)
        self.physx_interface.apply_body_force(body_path, drag_force, drag_torque)
        
        carb.log_info(f"Força Empuxo Aplicada: {buoyancy_force_magnitude:.2f} N")

    def start(self):
        """Configura a cena e subscreve ao evento de passo de física."""
        self.setup_scene()
        
        if self.physx_subscription is None:
            self.physx_subscription = self.physx_interface.subscribe_physics_step_events(self._on_physics_step)
            carb.log_info("Subscrição de física registada com sucesso. Pressione PLAY.")
        else:
            carb.log_warn("Subscrição de física já estava registada.")

    def stop(self):
        """Remove a subscrição de física para parar a aplicação de forças."""
        if self.physx_subscription is not None:
            self.physx_interface.unsubscribe_physics_step_events(self.physx_subscription)
            self.physx_subscription = None
            carb.log_info("Subscrição de física removida.")

# --- Ponto de Entrada do Script ---
if "buoyancy_manager" not in globals():
    buoyancy_manager = None

if buoyancy_manager is not None:
    buoyancy_manager.stop()

buoyancy_manager = BuoyancyInEditor()
buoyancy_manager.start()
