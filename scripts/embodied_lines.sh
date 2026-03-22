#!/bin/bash
# Count embodied framework lines (excluding channels/, cli/, skills/, templates/)
# Inspired by https://github.com/HKUDS/nanobot/blob/main/core_agent_lines.sh
cd "$(dirname "$0")/.." || exit 1

echo "roboclaw embodied framework line count"
echo "========================================="
echo ""

echo "  -- definition (what robots are) --"
for dir in definition/foundation definition/components definition/systems; do
  count=$(find "roboclaw/embodied/$dir" -name "*.py" -exec cat {} + 2>/dev/null | wc -l)
  printf "  %-40s %5s lines\n" "$dir/" "$count"
done

echo ""
echo "  -- execution (how robots move) --"
for dir in execution/integration/adapters execution/integration/carriers execution/integration/control_surfaces execution/integration/transports execution/orchestration/procedures execution/orchestration/runtime execution/observability; do
  count=$(find "roboclaw/embodied/$dir" -name "*.py" -exec cat {} + 2>/dev/null | wc -l)
  printf "  %-40s %5s lines\n" "$dir/" "$count"
done

echo ""
echo "  -- agent surface (tools + controller) --"
for file in execution/controller.py execution/tools.py; do
  count=$(cat "roboclaw/embodied/$file" 2>/dev/null | wc -l)
  printf "  %-40s %5s lines\n" "$file" "$count"
done

echo ""
echo "  -- onboarding + workspace --"
for dir in onboarding; do
  count=$(find "roboclaw/embodied/$dir" -name "*.py" -exec cat {} + 2>/dev/null | wc -l)
  printf "  %-40s %5s lines\n" "$dir/" "$count"
done
for file in workspace.py catalog.py localization.py probes.py; do
  count=$(cat "roboclaw/embodied/$file" 2>/dev/null | wc -l)
  printf "  %-40s %5s lines\n" "$file" "$count"
done

echo ""
echo "  -- builtins (per-embodiment declarations) --"
for file in roboclaw/embodied/builtins/*.py; do
  name=$(basename "$file")
  count=$(cat "$file" 2>/dev/null | wc -l)
  printf "  %-40s %5s lines\n" "builtins/$name" "$count"
done

echo ""
echo "  -- simulation (MuJoCo physics + ROS2 bridge) --"
for dir in simulation; do
  count=$(find "roboclaw/embodied/$dir" -name "*.py" -exec cat {} + 2>/dev/null | wc -l)
  printf "  %-40s %5s lines\n" "$dir/" "$count"
done

echo ""
echo "  -- perception (camera, streaming, snapshot) --"
for dir in perception; do
  count=$(find "roboclaw/embodied/$dir" -name "*.py" -exec cat {} + 2>/dev/null | wc -l)
  printf "  %-40s %5s lines\n" "$dir/" "$count"
done

echo ""
echo "  -- learning (data collection, training, policies) --"
for dir in learning; do
  count=$(find "roboclaw/embodied/$dir" -name "*.py" -exec cat {} + 2>/dev/null | wc -l)
  printf "  %-40s %5s lines\n" "$dir/" "$count"
done

echo ""
echo "  -- root --"
count=$(cat roboclaw/embodied/__init__.py 2>/dev/null | wc -l)
printf "  %-40s %5s lines\n" "__init__.py" "$count"

echo ""
echo "  ========================================="
total=$(find roboclaw/embodied -name "*.py" -exec cat {} + 2>/dev/null | wc -l)
printf "  Embodied total:                         %5s lines\n" "$total"
echo ""

# Also show the non-embodied agent core for context
agent_total=$(find roboclaw/agent -name "*.py" -exec cat {} + 2>/dev/null | wc -l)
printf "  Agent core (for comparison):            %5s lines\n" "$agent_total"
echo ""
echo "  (excludes: channels/, cli/, skills/, templates/)"
