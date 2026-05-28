from pathlib import Path
import re

# -----------------------------
# Patch clipseg_core.py
# -----------------------------
p = Path("clipseg_core.py")
s = p.read_text()

# Add prompt_cost_overrides argument
if "prompt_cost_overrides" not in s:
    s, n = re.subn(
        r'(debug_mode\s*:\s*str\s*=\s*["\']raw["\']\s*,\n)(\s*\):)',
        r'\1        prompt_cost_overrides: Optional[Dict[str, float]] = None,\n\2',
        s,
        count=1,
    )
    if n != 1:
        raise SystemExit("ERROR: cannot insert prompt_cost_overrides into ClipsegCore.__init__")

# Add override logic after self.prompt_costs assignment
if "Optional fixed prompt costs." not in s:
    s, n = re.subn(
        r'(        self\.prompt_costs\s*=\s*self\.build_prompt_costs\(self\.prompts,\s*self\.behavior_rule\)\n)',
        r'''\1
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

''',
        s,
        count=1,
    )
    if n != 1:
        raise SystemExit("ERROR: cannot find self.prompt_costs assignment in clipseg_core.py")

# Replace probs_to_cost_map
new_cost_func = r'''    def probs_to_cost_map(self, probs: np.ndarray) -> np.ndarray:
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

s, n = re.subn(
    r'    def probs_to_cost_map\(self, probs: np\.ndarray\) -> np\.ndarray:\n.*?(?=\n    @staticmethod\n    def make_)',
    new_cost_func,
    s,
    flags=re.S,
)
if n != 1:
    raise SystemExit(f"ERROR: replace probs_to_cost_map failed, replaced {n}")

# Add heatmap overlay function
heatmap_func = r'''
    @staticmethod
    def make_cost_heatmap_overlay(
        rgb: np.ndarray,
        cost_map: np.ndarray,
        alpha_max: float = 0.65,
        min_visible_cost: float = 1.0,
    ) -> np.ndarray:
        """
        Overlay semantic cost as a fixed-value navigation heatmap.

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
    s = s.replace(marker, heatmap_func + "\n" + marker)

# Use heatmap when debug_mode == cost
old = '''        if self.debug_mode == "cost":
            overlay_map = cost_map
        else:
            overlay_map = raw_prob_map

        debug_rgb = self.make_debug_overlay(rgb, overlay_map)
'''
new = '''        if self.debug_mode == "cost":
            debug_rgb = self.make_cost_heatmap_overlay(rgb, cost_map)
        else:
            debug_rgb = self.make_debug_overlay(rgb, raw_prob_map)
'''

if old in s:
    s = s.replace(old, new)
elif "debug_rgb = self.make_cost_heatmap_overlay(rgb, cost_map)" not in s:
    raise SystemExit("ERROR: cannot patch debug_rgb block")

p.write_text(s)
print("[OK] patched clipseg_core.py")


# -----------------------------
# Patch clipseg_debug_node.py
# -----------------------------
p = Path("clipseg_debug_node.py")
s = p.read_text()

s = s.replace("from typing import Optional, Tuple", "from typing import Dict, Optional, Tuple")

parser_func = r'''
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
    s = s.replace("\nclass ClipsegDebugNode(Node):", "\n" + parser_func + "\nclass ClipsegDebugNode(Node):")

if "prompt_cost_overrides=parse_prompt_cost_overrides(args.prompt_costs)" not in s:
    s, n = re.subn(
        r'(            debug_mode=args\.debug_mode,\n)(\s*\))',
        r'\1            prompt_cost_overrides=parse_prompt_cost_overrides(args.prompt_costs),\n\2',
        s,
        count=1,
    )
    if n != 1:
        raise SystemExit("ERROR: cannot insert prompt_cost_overrides into ClipsegCore call")

if "--prompt-costs" not in s:
    insert = r'''
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
    marker = '    parser.add_argument("--hz", type=float, default=0.5)\n'
    if marker not in s:
        raise SystemExit("ERROR: cannot find --hz marker for inserting --prompt-costs")
    s = s.replace(marker, insert + marker)

p.write_text(s)
print("[OK] patched clipseg_debug_node.py")


# -----------------------------
# Patch run_tita_mapping_system.sh
# -----------------------------
p = Path("run_tita_mapping_system.sh")
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
COST_STOP_GESTURE="${COST_STOP_GESTURE:-1.00}"'''
    )

s = s.replace("--debug-mode raw \\", "--debug-mode cost \\")
s = re.sub(r'--confidence-threshold\s+[0-9.]+\s+\\', '--confidence-threshold 0.25 \\\\', s)

if "--prompt-costs" not in s:
    s = s.replace(
        '    --prompts grass pavement vegetation "stop gesture" \\\n    --behavior-rule avoid_grass \\\n',
        '    --prompts grass pavement vegetation "stop gesture" \\\n    --prompt-costs "grass=$COST_GRASS" "pavement=$COST_PAVEMENT" "vegetation=$COST_VEGETATION" "stop gesture=$COST_STOP_GESTURE" \\\n    --behavior-rule avoid_grass \\\n'
    )

p.write_text(s)
print("[OK] patched run_tita_mapping_system.sh")
