import math
from typing import List, Optional, Tuple

import numpy as np
from pxr import Gf, UsdGeom, UsdPhysics, PhysicsSchemaTools, UsdUtils, Usd

import carb
import omni.kit.app as kit_app
import omni.physx
import omni.timeline
import omni.usd


class BuoyancySimulator:
    """A buoyancy simulator for a cube using the native PhysX API."""

    def __init__(
        self,
        cube_path: str = "/World/BuoyantCube",
        water_z: float = 0.0,
        cube_size: float = 1.0,
        cube_mass: float = 500.0,
        water_density: float = 1000.0,
        gravity: float = 9.81,
        linear_drag: float = 25.0,
        angular_drag: float = 2.0,
        air_damping: float = 0.1,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the buoyancy simulator.

        Parameters
        ----------
        cube_path : str
            The USD prim path where the cube will be created or fetched.
        water_z : float
            Height of the water plane in meters.  Points below this z value
            are considered submerged.
        cube_size : float
            Edge length of the cube in meters.
        cube_mass : float
            Mass of the cube in kilograms.
        water_density : float
            Density of the fluid (kg/m³).
        gravity : float
            Gravitational acceleration magnitude (m/s²).
        linear_drag : float
            Coefficient controlling linear drag when submerged.  Higher
            values result in stronger damping.
        angular_drag : float
            Coefficient controlling angular drag when submerged.
        air_damping : float
            Additional damping applied when the cube is fully above
            water to prevent jitter.
        verbose : bool
            Whether to print log messages via ``carb.log_info``.
        """
        self.cube_path = cube_path
        self.water_z = water_z
        self.size = cube_size
        self.mass = cube_mass
        self.rho = water_density
        self.g = gravity
        self.linear_drag_coeff = linear_drag
        self.angular_drag_coeff = angular_drag
        self.air_damping = air_damping
        self.verbose = verbose

        # Precompute corner offsets of the cube in local coordinates.  There
        # are 8 corners; for each we will compute buoyancy individually.
        half = 0.5 * self.size
        self.floaters_local: List[Gf.Vec3d] = [
            Gf.Vec3d(+half, +half, +half), Gf.Vec3d(+half, +half, -half),
            Gf.Vec3d(+half, -half, +half), Gf.Vec3d(+half, -half, -half),
            Gf.Vec3d(-half, +half, +half), Gf.Vec3d(-half, +half, -half),
            Gf.Vec3d(-half, -half, +half), Gf.Vec3d(-half, -half, -half),
        ]
        # Flat area associated with each floater (1/8th of the top face area).
        self.area_per_floater = (self.size * self.size) / 8.0

        # Stage and prim references (set in _setup_scene).
        self._stage = None
        self._cube_prim = None
        # PhysX simulation interface and identifiers set in _on_play.
        self._psi = None
        self._stage_id: Optional[int] = None
        self._body_id: Optional[int] = None

        # Previous pose used to compute velocities.
        self._prev_pos: Optional[np.ndarray] = None
        self._prev_quat: Optional[np.ndarray] = None
        self._prev_time: Optional[float] = None

        # Subscriptions and input state.
        self._timeline_sub = None
        self._update_sub = None
        self._input_sub = None
        self._is_running = False

        # Dragging state: when Shift + LMB is held, the cube becomes
        # kinematic and can be moved in the XY plane by the mouse.  The
        # variable ``_dragging`` tracks whether dragging is active and
        # ``_last_mouse`` stores the previous mouse position.
        self._dragging = False
        self._last_mouse: Optional[Tuple[float, float]] = None
        self._shift_down = False
        self._lmb_down = False
        # Mouse drag gain: meters per pixel per unit distance.  The
        # cube is moved by ``mouse_delta * mouse_gain * max(1.0, distance)``.
        self._mouse_gain = 0.002

        # Build the USD stage and scene objects.
        self._setup_scene()
        # Hook timeline and input events.
        self._subscribe_events()

        if self.verbose:
            carb.log_info("[BuoyancySimulator] Initialized. Press Play to start.")

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        """Create a USD stage with a physics scene, water plane and cube."""
        # Get the current stage from the USD context.
        self._stage = omni.usd.get_context().get_stage()
        # Ensure Z is the up axis for the stage.
        UsdGeom.SetStageUpAxis(self._stage, UsdGeom.Tokens.z)

        # Create the physics scene if it does not exist.  Gravity will be
        # handled automatically by PhysX; we set the magnitude and direction.
        scene_path = "/World/PhysicsScene"
        if not self._stage.GetPrimAtPath(scene_path):
            phys_scene_prim = UsdPhysics.Scene.Define(self._stage, scene_path).GetPrim()
            phys_scene = UsdPhysics.Scene(phys_scene_prim)
            # Set gravity direction downwards in Z.
            phys_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
            phys_scene.CreateGravityMagnitudeAttr().Set(self.g)

        # Create a water plane as a simple quad mesh for visual reference.
        water_path = "/World/WaterPlane"
        if not self._stage.GetPrimAtPath(water_path):
            mesh = UsdGeom.Mesh.Define(self._stage, water_path)
            extent = 50.0
            vertices = [
                Gf.Vec3f(-extent, -extent, self.water_z),
                Gf.Vec3f(extent, -extent, self.water_z),
                Gf.Vec3f(extent, extent, self.water_z),
                Gf.Vec3f(-extent, extent, self.water_z),
            ]
            mesh.CreatePointsAttr(vertices)
            mesh.CreateFaceVertexCountsAttr([4])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            # Normals and double-sided flag for proper shading.
            mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
            mesh.CreateDoubleSidedAttr(True)

        # Create the cube prim if it does not already exist.
        self._cube_prim = self._stage.GetPrimAtPath(self.cube_path)
        if not self._cube_prim or not self._cube_prim.IsValid():
            self._cube_prim = UsdGeom.Cube.Define(self._stage, self.cube_path).GetPrim()
            # Set size property (affects geometry but not transform).  Do not
            # scale the prim via transform; instead use the Cube schema's size.
            UsdGeom.Cube(self._cube_prim).CreateSizeAttr(self.size)
            # Apply RigidBody and Collision APIs to make it dynamic.
            UsdPhysics.RigidBodyAPI.Apply(self._cube_prim)
            UsdPhysics.CollisionAPI.Apply(self._cube_prim)
            # Assign a mass.
            mass_api = UsdPhysics.MassAPI.Apply(self._cube_prim)
            mass_api.CreateMassAttr().Set(self.mass)
            # Start with kinematic disabled so the body is dynamic.
            rb_api = UsdPhysics.RigidBodyAPI(self._cube_prim)
            rb_api.CreateKinematicEnabledAttr(False)
            # Add translate and orient transform ops if missing.
            xformable = UsdGeom.Xformable(self._cube_prim)
            ops = xformable.GetOrderedXformOps()
            has_translate = any(op.GetOpType() == UsdGeom.XformOp.TypeTranslate for op in ops)
            has_orient = any(op.GetOpType() == UsdGeom.XformOp.TypeOrient for op in ops)
            if not has_translate:
                translate_op = xformable.AddTranslateOp()
            else:
                translate_op = [op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate][0]
            if not has_orient:
                orient_op = xformable.AddOrientOp()
            else:
                orient_op = [op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeOrient][0]
            # Position the cube above the water surface.
            translate_op.Set(Gf.Vec3d(0.0, 0.0, self.water_z + 3.0))
            orient_op.Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # ------------------------------------------------------------------
    # Event subscription
    # ------------------------------------------------------------------
    def _subscribe_events(self) -> None:
        """Subscribe to timeline and input events."""
        app = kit_app.get_app()
        # Subscribe to input events for shift and mouse dragging.
        try:
            self._input_sub = app.get_input_event_stream().create_subscription_to_pop(self._on_input)
        except Exception:
            self._input_sub = None
        # Subscribe to timeline events (play/stop) to start/stop simulation.
        self.timeline = omni.timeline.get_timeline_interface()
        self._timeline_sub = self.timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline_event
        )

    # ------------------------------------------------------------------
    # Timeline event handling
    # ------------------------------------------------------------------
    def _on_timeline_event(self, event) -> None:
        """Handle play/stop events to attach/detach the physics simulation."""
        play_type = int(omni.timeline.TimelineEventType.PLAY)
        stop_type = int(omni.timeline.TimelineEventType.STOP)
        if event.type == play_type:
            if not self._is_running:
                # Initialize PhysX simulation interface and IDs.
                self._on_play()
        elif event.type == stop_type:
            if self._is_running:
                self._on_stop()

    def _on_play(self) -> None:
        """Called when the user presses Play.  Attach stage to PhysX and start updates."""
        # Acquire the PhysX simulation interface.
        self._psi = omni.physx.get_physx_simulation_interface()
        # Obtain stage_id via StageCache.  This ID is required by PhysX API.
        stage_cache = UsdUtils.StageCache.Get()
        self._stage_id = stage_cache.GetId(self._stage).ToLongInt()
        # Convert prim path to an integer actor ID using PhysicsSchemaTools.
        self._body_id = PhysicsSchemaTools.sdfPathToInt(self._cube_prim.GetPath())
        # Subscribe to the app update stream to apply forces every frame.
        app = kit_app.get_app()
        self._update_sub = app.get_update_event_stream().create_subscription_to_pop(self._on_update)
        # Initialize previous pose for velocity estimation.
        self._prev_pos = None
        self._prev_quat = None
        self._prev_time = None
        self._is_running = True
        carb.log_info("[BuoyancySimulator] Simulation started.")

    def _on_stop(self) -> None:
        """Called when the user presses Stop.  Detach stage and reset cube."""
        # Unsubscribe from update events.
        if self._update_sub is not None:
            self._update_sub.unsubscribe()
            self._update_sub = None
        self._is_running = False
        # Reset cube pose and velocity via USD transform.  To avoid applying
        # velocities through PhysX, temporarily make the body kinematic.
        rb_api = UsdPhysics.RigidBodyAPI(self._cube_prim)
        rb_api.GetKinematicEnabledAttr().Set(True)
        # Reset translation and orientation.
        xformable = UsdGeom.Xformable(self._cube_prim)
        translate_op = None
        orient_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                orient_op = op
        if translate_op:
            translate_op.Set(Gf.Vec3d(0.0, 0.0, self.water_z + 3.0))
        if orient_op:
            orient_op.Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        # Re-enable dynamic simulation.
        rb_api.GetKinematicEnabledAttr().Set(False)
        self._prev_pos = None
        self._prev_quat = None
        self._prev_time = None
        carb.log_info("[BuoyancySimulator] Simulation stopped and cube reset.")

    # ------------------------------------------------------------------
    # Input handling for dragging
    # ------------------------------------------------------------------
    def _on_input(self, event) -> None:
        """Handle input events for shift modifier and mouse dragging."""
        try:
            # Determine if shift or mouse buttons changed state.
            key = getattr(event, "input", None)
            val = float(getattr(event, "value", 0.0))
            is_pressed = val > 0.5
            if isinstance(key, str):
                k = key.upper()
                if "SHIFT" in k:
                    self._shift_down = is_pressed
                # Left mouse button.
                if k in ("MOUSE_LEFT_BUTTON", "LEFT_MOUSE_BUTTON", "MOUSE_BUTTON_LEFT"):
                    self._lmb_down = is_pressed
                    if not self._lmb_down:
                        # End dragging when button released.
                        self._dragging = False
                        self._last_mouse = None
                        # Restore dynamic simulation if we were dragging.
                        if self._is_running:
                            rb_api = UsdPhysics.RigidBodyAPI(self._cube_prim)
                            rb_api.GetKinematicEnabledAttr().Set(False)
            # Get mouse coordinates from event (fall back to position attr if present).
            mx = getattr(event, "mouseX", None)
            my = getattr(event, "mouseY", None)
            if mx is None or my is None:
                pos = getattr(event, "position", None)
                if pos and isinstance(pos, (tuple, list)) and len(pos) >= 2:
                    mx, my = float(pos[0]), float(pos[1])
            # Handle dragging if conditions met.
            if mx is not None and my is not None:
                self._handle_mouse(float(mx), float(my))
        except Exception:
            # Ignore input errors to avoid spamming logs.
            pass

    def _handle_mouse(self, mx: float, my: float) -> None:
        """Update dragging state and move the cube in the XY plane if dragging."""
        # Start dragging when Shift and LMB are pressed and any corner is submerged.
        if self._shift_down and self._lmb_down and self._body_partially_submerged():
            if not self._dragging:
                self._dragging = True
                self._last_mouse = (mx, my)
                # Make the body kinematic while dragging so we can reposition
                # without physics interfering.
                rb_api = UsdPhysics.RigidBodyAPI(self._cube_prim)
                rb_api.GetKinematicEnabledAttr().Set(True)
                return
            # If we were already dragging, compute displacement.
            if self._last_mouse is None:
                self._last_mouse = (mx, my)
                return
            dx = mx - self._last_mouse[0]
            dy = my - self._last_mouse[1]
            self._last_mouse = (mx, my)
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                return
            # Compute camera basis vectors to move along screen axes.
            right, up, distance = self._camera_basis()
            scale = self._mouse_gain * max(1.0, distance)
            dxy = right * dx * scale + up * (-dy) * scale
            # Move only in XY plane (no change in Z).
            dxy = Gf.Vec3d(dxy[0], dxy[1], 0.0)
            # Apply translation directly to the cube's transform.
            xformable = UsdGeom.Xformable(self._cube_prim)
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    current_t = op.Get()
                    new_t = Gf.Vec3d(current_t[0] + dxy[0], current_t[1] + dxy[1], current_t[2])
                    op.Set(new_t)
                    break
        else:
            # Not dragging; ensure state is reset.
            self._dragging = False
            self._last_mouse = None

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------
    def _body_partially_submerged(self) -> bool:
        """Return True if any corner of the cube is below the water plane."""
        # Compute current world pose.
        xformable = UsdGeom.Xformable(self._cube_prim)
        M = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        p = M.ExtractTranslation()
        # Extract rotation as a quaternion directly.  Using ExtractRotationQuat
        # avoids constructing a Gf.Quatd from a Gf.Rotation, which is invalid.
        q = M.ExtractRotationQuat()
        # Check each corner.
        for lp in self.floaters_local:
            wp = p + q.Transform(lp)
            if wp[2] < self.water_z:
                return True
        return False

    def _camera_basis(self) -> Tuple[Gf.Vec3d, Gf.Vec3d, float]:
        """Return camera right, up vectors and distance to the cube."""
        cam = None
        # Heuristic: look for a camera prim in the stage.  Common paths are
        # `/OmniverseKit_Persp/Camera`, `/OmniverseKit_Persp` and `/World/Camera`.
        for path in (
            "/OmniverseKit_Persp/Camera",
            "/OmniverseKit_Persp",
            "/World/Camera",
        ):
            prim = self._stage.GetPrimAtPath(path)
            if prim.IsValid() and prim.IsA(UsdGeom.Camera):
                cam = prim
                break
        if cam is None:
            # Find any camera in the stage by traversal.
            for prim in self._stage.Traverse():
                if prim.IsA(UsdGeom.Camera):
                    cam = prim
                    break
        if cam is None:
            # Default basis vectors and distance if no camera found.
            return Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), 5.0
        M = UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        # Right vector is first column of rotation matrix; up vector is second.
        right = Gf.Vec3d(M[0][0], M[1][0], M[2][0]).GetNormalized()
        up = Gf.Vec3d(M[0][1], M[1][1], M[2][1]).GetNormalized()
        cam_t = M.ExtractTranslation()
        p = UsdGeom.Xformable(self._cube_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        ).ExtractTranslation()
        dist = float(Gf.Vec3d(p[0] - cam_t[0], p[1] - cam_t[1], p[2] - cam_t[2]).GetLength())
        return right, up, max(0.5, dist)

    def _quat_to_numpy(self, q: Gf.Quatd) -> np.ndarray:
        """Convert a Gf.Quatd to a numpy array (w, x, y, z)."""
        return np.array([
            q.GetReal(),
            q.GetImaginary()[0],
            q.GetImaginary()[1],
            q.GetImaginary()[2],
        ], dtype=float)

    def _quat_inverse(self, q: np.ndarray) -> np.ndarray:
        """Return the inverse of quaternion q (w,x,y,z) assuming it is normalized."""
        w, x, y, z = q
        return np.array([w, -x, -y, -z], dtype=float)

    def _compute_angular_velocity(self, q_prev: np.ndarray, q_curr: np.ndarray, dt: float) -> np.ndarray:
        """Estimate angular velocity from quaternion difference over dt.

        The angular velocity vector ω satisfies dq/dt = 0.5 * ω ⊗ q (where ⊗ is quaternion product).
        A finite difference approximation gives ω = 2 * (dq * q_prev^{-1}).vector / dt, where
        dq = q_curr * q_prev^{-1}.  The vector part of dq (x,y,z) scaled by the scalar part gives the
        rotation axis multiplied by sin(θ/2), and the scalar part gives cos(θ/2).  For small angles
        this yields a good approximation.
        """
        # Compute delta quaternion dq = q_curr * inverse(q_prev)
        q_inv = self._quat_inverse(q_prev)
        # Quaternion multiplication (w1, v1) * (w2, v2) = (w1*w2 - v1·v2, w1*v2 + w2*v1 + v1×v2)
        w1, v1 = q_curr[0], q_curr[1:]
        w2, v2 = q_inv[0], q_inv[1:]
        dq_w = w1 * w2 - np.dot(v1, v2)
        dq_v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
        # Avoid division by zero when dt is tiny.
        if dt <= 1e-8:
            return np.zeros(3, dtype=float)
        # Angular velocity: ω = 2 * dq_v / dt (for small angles assume dq_w ≈ 1).
        return (2.0 / dt) * dq_v

    # ------------------------------------------------------------------
    # Physics update
    # ------------------------------------------------------------------
    def _on_update(self, event) -> None:
        """Compute buoyancy and drag forces and apply them via PhysX."""
        # Ensure we have valid psi and IDs.
        if not self._psi or self._stage_id is None or self._body_id is None:
            return
        # Compute time step using the timeline.
        current_time = float(self.timeline.get_current_time())
        if self._prev_pos is None or self._prev_quat is None or self._prev_time is None:
            # Initialize previous pose; do not apply forces this frame.
            xformable = UsdGeom.Xformable(self._cube_prim)
            M = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            p = M.ExtractTranslation()
            # Extract rotation quaternion directly.  This returns a Gf.Quatd or Gf.Quatf
            # and avoids constructing a quaternion from a Gf.Rotation.
            q = self._quat_to_numpy(M.ExtractRotationQuat())
            self._prev_pos = np.array([p[0], p[1], p[2]], dtype=float)
            self._prev_quat = q
            self._prev_time = current_time
            return
        dt = current_time - self._prev_time
        if dt <= 0.0:
            self._prev_time = current_time
            return
        # Read current pose of the cube.
        xformable = UsdGeom.Xformable(self._cube_prim)
        M = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        p = M.ExtractTranslation()
        # Extract rotation quaternion directly.
        q_curr = self._quat_to_numpy(M.ExtractRotationQuat())
        p_curr = np.array([p[0], p[1], p[2]], dtype=float)
        # Compute linear velocity by finite difference.
        v = (p_curr - self._prev_pos) / dt
        # Compute angular velocity vector.
        w = self._compute_angular_velocity(self._prev_quat, q_curr, dt)

        # Compute buoyancy and drag forces.
        total_force = np.zeros(3, dtype=float)
        total_torque = np.zeros(3, dtype=float)
        fraction_sum = 0.0
        q_obj = Gf.Quatd(q_curr[0], Gf.Vec3d(q_curr[1], q_curr[2], q_curr[3]))
        for lp in self.floaters_local:
            # Transform local point to world coordinates.
            world_offset = q_obj.Transform(lp)
            wp = Gf.Vec3d(p_curr[0] + world_offset[0], p_curr[1] + world_offset[1], p_curr[2] + world_offset[2])
            depth = self.water_z - wp[2]
            if depth > 0.0:
                # Fraction of cube height submerged at this corner.
                ratio = min(depth / self.size, 1.0)
                fraction_sum += ratio
                # Buoyant force magnitude: ρ g area min(depth, size).
                fb = self.rho * self.g * self.area_per_floater * min(depth, self.size)
                f_vec = np.array([0.0, 0.0, fb], dtype=float)
                total_force += f_vec
                # Torque due to buoyant force: r × f.
                r_vec = np.array([world_offset[0], world_offset[1], world_offset[2]], dtype=float)
                total_torque += np.cross(r_vec, f_vec)
        # Compute submerged mass fraction (average ratio across corners).
        fraction_submerged = fraction_sum / len(self.floaters_local)
        partial_mass = self.mass * fraction_submerged
        if partial_mass > 0.0:
            # Linear drag opposing vertical velocity.
            drag_force_z = -self.linear_drag_coeff * partial_mass * v[2]
            total_force += np.array([0.0, 0.0, drag_force_z], dtype=float)
            # Angular drag opposing angular velocity.  Characteristic length squared is (0.5 * size)^2.
            L_avg_sq = (0.5 * self.size) ** 2
            total_torque += -partial_mass * self.angular_drag_coeff * L_avg_sq * w
        else:
            # Apply small damping in air for stability.
            total_force += -self.air_damping * self.mass * v
            total_torque += -self.air_damping * self.mass * w
        # Convert forces and torques to carb.Float3 and apply at centre of mass.
        force_carb = carb._carb.Float3(total_force[0], total_force[1], total_force[2])
        pos_carb = carb._carb.Float3(p_curr[0], p_curr[1], p_curr[2])
        torque_carb = carb._carb.Float3(total_torque[0], total_torque[1], total_torque[2])
        # Wake up the rigid body to ensure forces take effect.
        try:
            self._psi.wake_up(self._stage_id, self._body_id)
        except Exception:
            pass
        # Apply force and torque.  Use default mode ("Force").
        self._psi.apply_force_at_pos(self._stage_id, self._body_id, force_carb, pos_carb)
        self._psi.apply_torque(self._stage_id, self._body_id, torque_carb)
        # Store current pose for next frame.
        self._prev_pos = p_curr.copy()
        self._prev_quat = q_curr.copy()
        self._prev_time = current_time


def main() -> None:
    """Entry point to run the buoyancy simulator in Isaac Sim."""
    BuoyancySimulator()


if __name__ == "__main__":
    main()
