from pathlib import Path
import re

def backup_once(path: str):
    p = Path(path)
    b = Path(path + ".bak_fixed_prompt_cost_exact")
    if p.exists() and not b.exists():
        b.write_text(p.read_text())

backup_once("clipseg_core.py")
backup_once("clipseg_debug_node.py")
backup_once("run_tita_mapping_system.sh")

# ============================================================
# 1. Patch clipseg_core.py
# ============================================================
p = Path("clipseg_core.py")
s = p.read_text()

# Add prompt_cost_overrides to __init__
if "prompt_cost_overrides" not in s:
    old = '''        debug_mode: str = "raw",
        make_debug_overlay: bool = True,
    ):'''
    new = '''        debug_mode: str = "raw",
        make_debug_overlay: bool = True,
        prompt_cost_overrides: Optional[Dict[str, float]] = None,
    ):'''
    if old not in s:
        raise SystemExit("ERROR: cannot find ClipsegCore __init__ signature block")
    s = s.replace(old, new, 1)

# Apply fixed prompt costs after default prompt_costs
if "Optional fixed prompt costs." not in s:
    old = '''        self.prompt_costs = self.build_prompt_costs(self.prompts, self.behavior_rule)
'''
    new = '''        self.prompt_costs = self.build_prompt_costs(self.prompts, self.behavior_rule)

        # Optional fixed prompt costs.
        # Values can be normalized [0,1], e.g. 0.78,
        # or OccupancyGrid-style [0,100], e.g. 78.
        if prompt_cost_overrides:
            for key, value in prompt_cost_overrides.items():
                k = str(key).strip().lower()
                v = float(value)
                if v <= 1.0:
                    v *= 100.0
                v = float(np.clip(v, 0.0, 100.0))

                for prompt in self.prompts:
                    if prompt.strip().lower() == k:
                        self.prompt_costs[prompt] = v

'''
    if old not in s:
        raise SystemExit("ERROR: cannot find self.prompt_costs assignment")
    s = s.replace(old, new, 1)

# Replace probs_to_cost_map with fixed-cost logic
new_cost_func = '''    def probs_to_cost_map(self, probs: np.ndarray) -> np.ndarray:
        """
        Convert per-prompt probability maps to a fixed-cost semantic map.

        For each prompt:
          if prob >= confidence_threshold:
              pixel cost = fixed prompt cost

        If multiple prompts are active on the same pixel,
        the highest configured cost wins.
        """
        if probs.ndim != 3:
            raise ValueError(f"Expected CxHxW probs, got shape={probs.shape}")

        c, h, w = probs.shape
        cost_map = np.zeros((h, w), dtype=np.float32)

        for i, prompt in enumerate(self.prompts):
            if i >= c:
                break

            prompt_cost = float(self.prompt_costs.get(prompt, 0.0))
            if prompt_cost <= 0.0:
                continue

            active = probs[i] >= self.confidence_threshold
            if np.any(active):
                cost_map[active] = np.maximum(cost_map[active], prompt_cost)

        return np.clip(cost_map, 0, 100).astype(np.uint8)

'''

start = s.find("    def probs_to_cost_map(")
if start < 0:
    raise SystemExit("ERROR: cannot find probs_to_cost_map")

end = s.find("\n    @staticmethod\n    def make_", start)
if end < 0:
    raise SystemExit("ERROR: cannot find end of probs_to_cost_map")

s = s[:start] + new_cost_func + s[end+1:]

# Add heatmap overlay function
heatmap_func = '''
    @staticmethod
    def make_cost_heatmap_overlay(
        rgb: np.ndarray,
        cost_map: np.ndarray,
        alpha_max: float = 0.65,
        min_visible_cost: float = 1.0,
    ) -> np.ndarray:
        """
        Overlay fixed semantic cost as a navigation heatmap.

        cost 0      : no overlay
        cost 1-50   : green -> yellow
        cost 50-100 : yellow -> red
        """
        h, w = rgb.shape[:2]

        if cost_map.shape != (h, w):
            img = PILImage.fromarray(cost_map.astype(np.uint8), mode="L")
            cost_map = np.asarray(
                img.resize((w, h), resample=PILImage.BILINEAR),
                dtype=np.uint8,
            )

        rgb_f = rgb.astype(np.float32)
        c = np.clip(cost_map.astype(np.float32) / 100.0, 0.0, 1.0)

        heat = np.zeros_like(rgb_f)

        low = c < 0.5
        high = ~low

        r = heat[..., 0]
        g = heat[..., 1]
        b = heat[..., 2]

        # 0.0 -> green, 0.5 -> yellow
        r[low] = 255.0 * (2.0 * c[low])
        g[low] = 255.0
        b[low] = 0.0

        # 0.5 -> yellow, 1.0 -> red
        r[high] = 255.0
        g[high] = 255.0 * (2.0 - 2.0 * c[high])
        b[high] = 0.0

        valid = cost_map.astype(np.float32) >= float(min_visible_cost)
        alpha = (alpha_max * c)[..., None]

        out = rgb_f.copy()
        if np.any(valid):
            out[valid] = (
                (1.0 - alpha[valid]) * rgb_f[valid]
                + alpha[valid] * heat[valid]
            )

        return np.clip(out, 0, 255).astype(np.uint8)

'''

if "def make_cost_heatmap_overlay" not in s:
    marker = "    @staticmethod\n    def make_debug_overlay("
    if marker not in s:
        raise SystemExit("ERROR: cannot find make_debug_overlay marker")
    s = s.replace(marker, heatmap_func + "\n" + marker, 1)

# Use fixed-cost heatmap when debug_mode == cost
old = '''        if self.make_debug_overlay_enabled:
            overlay_map = cost_map if self.debug_mode == "cost" else raw_prob_map
            debug_rgb = self.make_debug_overlay(rgb, overlay_map)
        else:
            debug_rgb = None
'''
new = '''        if self.make_debug_overlay_enabled:
            if self.debug_mode == "cost":
                debug_rgb = self.make_cost_heatmap_overlay(rgb, cost_map)
            else:
                debug_rgb = self.make_debug_overlay(rgb, raw_prob_map)
        else:
            debug_rgb = None
'''
if old in s:
    s = s.replace(old, new, 1)
elif "debug_rgb = self.make_cost_heatmap_overlay(rgb, cost_map)" not in s:
    raise SystemExit("ERROR: cannot patch debug overlay block")

p.write_text(s)
print("[OK] patched clipseg_core.py")


# ============================================================
# 2. Patch clipseg_debug_node.py
# ============================================================
p = Path("clipseg_debug_node.py")
s = p.read_text()

s = s.replace(
    "from typing import Optional, Tuple",
    "from typing import Dict, Optional, Tuple",
)

parser_func = '''
def parse_prompt_cost_overrides(items) -> Dict[str, float]:
    """
    Parse prompt cost overrides.

    Examples:
      grass=0.56
      pavement=0.05
      vegetation=0.78
      "stop gesture=1.00"

    Values can be [0,1] or [0,100].
    """
    out: Dict[str, float] = {}
    if not items:
        return out

    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --prompt-costs item '{item}'. Expected format: name=value"
            )

        name, value = item.split("=", 1)
        name = name.strip().lower()
        value = float(value.strip())

        if not name:
            raise ValueError(f"Invalid empty prompt name in --prompt-costs item '{item}'")

        out[name] = value

    return out


'''

if "def parse_prompt_cost_overrides" not in s:
    marker = "\nclass ClipsegDebugNode(Node):"
    if marker not in s:
        raise SystemExit("ERROR: cannot find ClipsegDebugNode class marker")
    s = s.replace(marker, "\n" + parser_func + marker, 1)

# Pass fixed costs into ClipsegCore
if "prompt_cost_overrides=parse_prompt_cost_overrides(args.prompt_costs)" not in s:
    old = '''            debug_mode=args.debug_mode,
            make_debug_overlay=args.publish_debug_image,
        )
'''
    new = '''            debug_mode=args.debug_mode,
            make_debug_overlay=args.publish_debug_image,
            prompt_cost_overrides=parse_prompt_cost_overrides(args.prompt_costs),
        )
'''
    if old not in s:
        raise SystemExit("ERROR: cannot find ClipsegCore constructor block")
    s = s.replace(old, new, 1)

# Add --prompt-costs arg
if "--prompt-costs" not in s:
    old = '''    parser.add_argument(
        "--behavior-rule",
        default="avoid_grass",
        choices=["avoid_grass", "allow_grass", "prefer_pavement"],
    )

'''
    new = '''    parser.add_argument(
        "--behavior-rule",
        default="avoid_grass",
        choices=["avoid_grass", "allow_grass", "prefer_pavement"],
    )

    parser.add_argument(
        "--prompt-costs",
        nargs="*",
        default=None,
        help=(
            "Fixed semantic cost per prompt. "
            "Examples: grass=0.56 pavement=0.05 vegetation=0.78 'stop gesture=1.00'. "
            "Values can be normalized [0,1] or cost [0,100]."
        ),
    )

'''
    if old not in s:
        raise SystemExit("ERROR: cannot find behavior-rule argument block")
    s = s.replace(old, new, 1)

# Better defaults
s = s.replace(
    'parser.add_argument("--confidence-threshold", type=float, default=0.05)',
    'parser.add_argument("--confidence-threshold", type=float, default=0.25)',
)
s = s.replace(
    'parser.add_argument("--debug-mode", choices=["raw", "cost"], default="raw")',
    'parser.add_argument("--debug-mode", choices=["raw", "cost"], default="cost")',
)

p.write_text(s)
print("[OK] patched clipseg_debug_node.py")


# ============================================================
# 3. Patch run_tita_mapping_system.sh
# ============================================================
p = Path("run_tita_mapping_system.sh")
if p.exists():
    s = p.read_text()

    if "COST_GRASS" not in s:
        s = s.replace(
            'PUBLISH_DEBUG_IMAGE="${PUBLISH_DEBUG_IMAGE:-0}"',
            '''PUBLISH_DEBUG_IMAGE="${PUBLISH_DEBUG_IMAGE:-0}"

# Fixed semantic costs for each CLIPSeg prompt.
# Values can be [0,1]. They are converted to [0,100] internally.
COST_GRASS="${COST_GRASS:-0.56}"
COST_PAVEMENT="${COST_PAVEMENT:-0.05}"
COST_VEGETATION="${COST_VEGETATION:-0.78}"
COST_STOP_GESTURE="${COST_STOP_GESTURE:-1.00}"''',
            1,
        )

    lines = s.splitlines()
    out = []
    has_prompt_costs = "--prompt-costs" in s

    for line in lines:
        stripped = line.strip()
        indent = line[:len(line) - len(line.lstrip())]

        if stripped.startswith("--debug-mode "):
            line = indent + "--debug-mode cost \\"
        elif stripped.startswith("--confidence-threshold "):
            line = indent + "--confidence-threshold 0.25 \\"

        out.append(line)

        if (
            not has_prompt_costs
            and "--prompts " in stripped
            and "grass" in stripped
            and "pavement" in stripped
            and "vegetation" in stripped
            and "stop gesture" in stripped
        ):
            out.append(
                indent
                + '--prompt-costs "grass=$COST_GRASS" "pavement=$COST_PAVEMENT" '
                + '"vegetation=$COST_VEGETATION" "stop gesture=$COST_STOP_GESTURE" \\'
            )
            has_prompt_costs = True

    s = "\n".join(out) + "\n"
    p.write_text(s)
    print("[OK] patched run_tita_mapping_system.sh")
else:
    print("[WARN] run_tita_mapping_system.sh not found, skipped")
