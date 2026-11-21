// Room-by-room push-optimal Sokoban solver.
//
// The levels this competition draws are a 1-wide hallway plus 1-4 rooms
// off it, each room holding 3-4 boxes and as many goals. A box pushed out
// of a room through its door into the hallway can never come back, so every
// box stays in its own room and the level's push optimum P* is the sum of
// the rooms' optima. This solver therefore splits each level into rooms,
// solves every room push-optimally on its own, and stitches the room
// solutions together with shortest walks between pushes.
//
// Each room is solved by breadth-first search over pushes (so the first
// solved state found is push-optimal). A push state is the boxes' squares
// plus the player's reachable region, named by its lowest square. Two
// sound prunings keep the search small: a box may only be pushed onto a
// "live" square (one from which a lone box can still be pulled back onto
// some goal), and a push that leaves a box frozen inside a 2x2 block of
// walls and boxes with a box off its goal is discarded. Neither pruning
// removes a solvable state, so optimality stands.
//
// Everything is budgeted by work (states expanded), never by the clock, so
// two runs on one set produce byte-identical output.
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets (states), never wall-clock. The generator only keeps a room
// its own push-optimal search solves within 250k states, so 300k per room
// with one larger retry leaves ample headroom inside the 60s wall budget.
const ROOM_STATES = 300_000;
const ROOM_RETRY_STATES = 1_200_000;
const WHOLE_STATES = 400_000;
const GLOBAL_STATES = 12_000_000;

let spent = 0; // states expanded across the whole run

// Directions in the order the tie-breaks use: up, down, left, right.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const STEP = ['u', 'd', 'l', 'r'];
const PUSH = ['U', 'D', 'L', 'R'];

interface Board {
  w: number;
  h: number;
  wall: Uint8Array; // 1 = wall
  goals: number[];
  goalSet: Uint8Array;
  boxes: number[]; // sorted
  player: number;
}

function parseLevel(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  const goalSet = new Uint8Array(w * h);
  for (const g of goals) goalSet[g] = 1;
  return { w, h, wall, goals, goalSet, boxes, player };
}

// Squares from which a lone box can be pulled back onto some goal: start
// at the goals and pull backwards (a box on x could have come from x+d if
// the player could stand at x-d, i.e. both x-d and x-2d are open floor).
function liveSquares(b: Board): Uint8Array {
  const live = new Uint8Array(b.w * b.h);
  const queue: number[] = [];
  for (const g of b.goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (let k = 0; k < 4; k++) {
      const d = DY[k]! * b.w + DX[k]!;
      const from = x - d;
      const stand = from - d;
      if (from < 0 || stand < 0 || from >= live.length || stand >= live.length) continue;
      if (!b.wall[from] && !b.wall[stand] && !live[from]) {
        live[from] = 1;
        queue.push(from);
      }
    }
  }
  return live;
}

// Flood-fill the player's reachable region from `from` with `occ` marking
// boxes; stamps the region in `seen` with `stamp` and returns its lowest
// square. `stack` is scratch space.
function flood(
  b: Board, occ: Uint8Array, from: number, seen: Int32Array, stamp: number, stack: Int32Array,
): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k]! * b.w + DX[k]!;
      if (q < 0 || q >= seen.length) continue;
      if (!b.wall[q] && !occ[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

// True when the box just pushed to `t` sits in a frozen 2x2 block: four
// cells all wall-or-box, with a box among them off its goal.
function frozenBlock(b: Board, occ: Uint8Array, t: number): boolean {
  const w = b.w;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= occ.length) continue;
    if (!(b.wall[a] || occ[a]) || !(b.wall[a + 1] || occ[a + 1])) continue;
    if (!(b.wall[c] || occ[c]) || !(b.wall[d] || occ[d])) continue;
    if ((occ[a] && !b.goalSet[a]) || (occ[a + 1] && !b.goalSet[a + 1]) ||
        (occ[c] && !b.goalSet[c]) || (occ[d] && !b.goalSet[d])) return true;
  }
  return false;
}

// Shortest walk from `from` to `to` around walls and boxes (lowercase LURD,
// first direction first on ties). Throws if unreachable.
function walkTo(b: Board, occ: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(b.w * b.h).fill(-1);
  const how = new Array<string>(b.w * b.h);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k]! * b.w + DX[k]!;
      if (q < 0 || q >= prev.length) continue;
      if (b.wall[q] || occ[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = STEP[k]!;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walk target unreachable');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

interface Solved {
  pushes: number;
  moves: string; // lowercase steps, uppercase pushes
}

// Push-optimal BFS for one board. Returns null when the budget runs out or
// the board is unsolvable. Deterministic: fixed expansion order, no clock.
function solvePushOptimal(b: Board, maxStates: number): Solved | null {
  const n = b.w * b.h;
  const size = n;
  const live = liveSquares(b);
  if (b.boxes.some((x) => !live[x])) return null;
  const occ = new Uint8Array(n);
  const seen = new Int32Array(n);
  const mark = new Int32Array(n); // child keys flood here, never over `seen`:
  // a key flood re-stamping the current state's region would make later
  // pushes of the same state look unreachable and wrongly prune them.
  const stack = new Int32Array(n);
  let stamp = 0;
  const solved = (bs: number[]): boolean => bs.every((x) => b.goalSet[x]);

  // State key: region-low, then the boxes, as digits in base `size`.
  const keyOf = (bs: number[], low: number): number => {
    let k = low;
    for (const x of bs) k = k * size + x;
    return k;
  };

  const boxesArr: number[][] = [b.boxes.slice()];
  const playerArr: number[] = [b.player];
  const parent: number[] = [-1];
  const pushBox: number[] = [-1]; // box square before the push that made this state
  const pushDir: number[] = [-1];
  const index = new Map<number, number>();
  for (const x of b.boxes) occ[x] = 1;
  const startLow = flood(b, occ, b.player, seen, ++stamp, stack);
  for (const x of b.boxes) occ[x] = 0;
  index.set(keyOf(b.boxes, startLow), 0);
  if (solved(b.boxes)) return { pushes: 0, moves: '' };

  let found = -1;
  let expanded = 0;
  for (let head = 0; found < 0 && head < boxesArr.length; head++) {
    const boxes = boxesArr[head]!;
    for (const x of boxes) occ[x] = 1;
    const here = ++stamp;
    flood(b, occ, playerArr[head]!, seen, here, stack);
    expanded++;
    spent++;
    if (expanded > maxStates || spent > GLOBAL_STATES) {
      for (const x of boxes) occ[x] = 0;
      return null;
    }
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const box = boxes[bi]!;
      for (let k = 0; k < 4; k++) {
        const d = DY[k]! * b.w + DX[k]!;
        const dest = box + d;
        const stand = box - d;
        if (stand < 0 || dest < 0 || stand >= n || dest >= n) continue;
        // The player must be able to stand behind the box to push it, and
        // the destination must be open floor, live, and unoccupied.
        if (seen[stand] !== here || b.wall[dest] || occ[dest] || !live[dest]) continue;
        occ[box] = 0;
        occ[dest] = 1;
        const frozen = frozenBlock(b, occ, dest);
        let key = -1;
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = dest;
          next.sort((a, c) => a - c);
          key = keyOf(next, flood(b, occ, box, mark, ++stamp, stack));
        }
        occ[dest] = 0;
        occ[box] = 1;
        if (frozen || index.has(key)) continue;
        if (boxesArr.length >= maxStates) {
          for (const x of boxes) occ[x] = 0;
          return null;
        }
        index.set(key, boxesArr.length);
        boxesArr.push(next!);
        playerArr.push(box);
        parent.push(head);
        pushBox.push(box);
        pushDir.push(k);
        if (solved(next!)) {
          found = boxesArr.length - 1;
          break;
        }
      }
    }
    for (const x of boxes) occ[x] = 0;
  }
  if (found < 0) return null;

  // Rebuild the move string: walk to each push square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = b.player;
  let boxes = b.boxes.slice();
  for (const s of path) {
    const nextBoxes = boxesArr[s]!;
    const from = pushBox[s]!;
    const k = pushDir[s]!;
    const d = DY[k]! * b.w + DX[k]!;
    for (const x of boxes) occ[x] = 1;
    moves += walkTo(b, occ, player, from - d);
    for (const x of boxes) occ[x] = 0;
    moves += PUSH[k]!;
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

// The square the player ends on after `moves` (case-insensitive LURD).
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) {
    p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  }
  return p;
}

interface Room {
  boxes: number[];
  goals: number[];
}

// Split a level into rooms: cut the dead squares of the player's row (the
// hallway: a box there reaches no goal) out of the floor; each remaining
// connected part holding boxes is a room. Null when the level is not such
// a set of rooms.
function roomsOf(b: Board): Room[] | null {
  const live = liveSquares(b);
  const row = Math.floor(b.player / b.w);
  const cut = new Uint8Array(b.w * b.h);
  for (let i = row * b.w; i < (row + 1) * b.w; i++) {
    if (!b.wall[i] && !live[i]) cut[i] = 1;
  }
  if (b.boxes.some((x) => cut[x])) return null;
  const part = new Int32Array(b.w * b.h).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < b.w * b.h; s++) {
    if (b.wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (let k = 0; k < 4; k++) {
        const q = squares[i]! + DY[k]! * b.w + DX[k]!;
        if (q < 0 || q >= part.length) continue;
        if (!b.wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [] });
  }
  for (const x of b.boxes) rooms[part[x]!]!.boxes.push(x);
  for (const g of b.goals) {
    if (part[g]! < 0) return null;
    rooms[part[g]!]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

// Solve one level room by room; fall back to a whole-level search when the
// level is not a set of rooms, or when a room's search gives up.
function solveLevel(text: string): string {
  const b = parseLevel(text);
  if (b.boxes.every((x) => b.goalSet[x])) return '';
  const rooms = roomsOf(b);
  if (rooms !== null) {
    let player = b.player;
    let moves = '';
    const done: boolean[] = rooms.map(() => false);
    for (let ri = 0; ri < rooms.length; ri++) {
      const room = rooms[ri]!;
      const wall = b.wall.slice();
      for (let oj = 0; oj < rooms.length; oj++) {
        if (oj === ri) continue;
        const other = rooms[oj]!;
        // Other rooms' boxes stand as walls; a finished room's boxes sit
        // on its goals, which is the same set of squares either way.
        for (const x of done[oj]! ? other.goals : other.boxes) wall[x] = 1;
      }
      const sub: Board = {
        w: b.w, h: b.h, wall,
        goals: room.goals.slice().sort((a, c) => a - c),
        goalSet: b.goalSet,
        boxes: room.boxes.slice().sort((a, c) => a - c),
        player,
      };
      // The sub-board's goal set is the whole level's, but only this
      // room's goals are targets; boxes already on goals still count.
      let solved = solvePushOptimal(sub, ROOM_STATES);
      if (solved === null) solved = solvePushOptimal(sub, ROOM_RETRY_STATES);
      if (solved === null) return '';
      moves += solved.moves;
      player = endOf(b.w, player, solved.moves);
      done[ri] = true;
    }
    return moves.toLowerCase();
  }
  const whole = solvePushOptimal(b, WHOLE_STATES);
  return whole === null ? '' : whole.moves.toLowerCase();
}

for (const id of readdirSync(I).sort()) {
  // Skip anything that is not an instance directory (e.g. a seed file).
  if (!existsSync(join(I, id, 'level.txt'))) continue;
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
