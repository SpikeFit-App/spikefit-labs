// SpikeFit Labs: Approach Analyzer scoring engine.
//
// Pure functions only. Input is a list of per-frame pose landmarks (pixels),
// output is detected approach events plus a technique checklist. Nothing here
// touches the network, the DOM, or storage.
//
// Technique targets come from two coaching references:
//   - Intent step and hip height: lower your hips on the intent step (the step
//     before the penultimate), then hold that height through the penultimate
//     step instead of sinking right before takeoff.
//   - 3-step approach: 20-30 degree forward lean at the start, step lengths
//     medium / long / short, tempo slow to fast, arms forward-back-up with no
//     "scissored" arms, slight lean at the plant.
//
// Every number below is a starting guess for tuning, not a published norm.
// Distances are in leg lengths (hip to knee to ankle) so results don't depend
// on how far the camera is from the athlete.

export const CONFIG = {
    minVisibility: 0.5,          // landmark visibility needed to trust a point
    farArmVisibility: 0.8,       // side view misreads a half-hidden far arm, so only trust it when clearly seen
    smoothWindow: 5,             // frames in the moving-average filter (odd)
    minPoseCoverage: 0.6,        // share of frames that must have a pose

    contactHeight: 0.15,         // foot within this many leg lengths of the floor (heel-first plants sit high)
    contactSpeed: 1.5,           // and moving slower than this (leg lengths/s)
    minContactSec: 0.04,         // shortest foot contact that counts as a plant
    mergePlantDistance: 0.15,    // contacts closer than this are the same plant
    motionStartSpeed: 0.25,      // hip speed (leg lengths/s) that marks the start
    minJumpRise: 0.10,           // hip rise after takeoff needed to call it a jump

    startLeanMin: 20,            // starting stance forward lean, degrees
    startLeanMax: 30,
    startLeanSlack: 8,           // degrees outside the range that still only warns
    approachLeanMin: 5,          // chest ahead of hips during the approach
    plantLeanMax: 30,            // "slight lean" at the plant

    intentHipDropMin: 0.05,      // hips should drop at least this on the intent step
    penultimateSinkMax: 0.04,    // and sink no more than this on the penultimate

    tempoSlack: 1.10,            // a later step may be up to 10% slower and still pass

    armBackMin: 0.20,            // wrist this far behind the shoulder counts as "back"
    armBackEarlySec: 0.15,       // backswing may peak this long before the penultimate plant
    straightElbowMin: 140,       // elbow angle at full backswing, degrees
    straightElbowWarn: 120,
    scissorGap: 0.40,            // wrists this far apart, on opposite sides, = scissored
    scissorWarn: 0.25,           // share of frames scissored before warning
    scissorFail: 0.50,
};

const L = { sh: 11, el: 13, wr: 15, hip: 23, kn: 25, an: 27, he: 29, toe: 31 };
const R = { sh: 12, el: 14, wr: 16, hip: 24, kn: 26, an: 28, he: 30, toe: 32 };
const SIDES = { left: L, right: R };

// ---------- small math helpers ----------

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, v: Math.min(a.v, b.v) });

function median(values) {
    const v = values.filter(Number.isFinite).sort((a, b) => a - b);
    if (!v.length) return NaN;
    const m = Math.floor(v.length / 2);
    return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

function percentile(values, p) {
    const v = values.filter(Number.isFinite).sort((a, b) => a - b);
    if (!v.length) return NaN;
    const i = Math.min(v.length - 1, Math.max(0, Math.round((p / 100) * (v.length - 1))));
    return v[i];
}

function angleDeg(a, b, c) {
    // Angle at b formed by a-b-c.
    const v1 = { x: a.x - b.x, y: a.y - b.y };
    const v2 = { x: c.x - b.x, y: c.y - b.y };
    const d = Math.hypot(v1.x, v1.y) * Math.hypot(v2.x, v2.y);
    if (!d) return NaN;
    const cos = Math.max(-1, Math.min(1, (v1.x * v2.x + v1.y * v2.y) / d));
    return (Math.acos(cos) * 180) / Math.PI;
}

const round = (n, d = 2) => (Number.isFinite(n) ? Number(n.toFixed(d)) : null);

// ---------- preprocessing ----------

// Moving average per landmark. Frames without a pose stay null.
function smoothFrames(frames, window) {
    const half = Math.floor(window / 2);
    return frames.map((frame, i) => {
        if (!frame.lm) return { t: frame.t, lm: null };
        const lm = frame.lm.map((_, k) => {
            let sx = 0, sy = 0, sv = 0, n = 0;
            for (let j = Math.max(0, i - half); j <= Math.min(frames.length - 1, i + half); j++) {
                const p = frames[j].lm && frames[j].lm[k];
                if (!p) continue;
                sx += p.x; sy += p.y; sv += p.v; n++;
            }
            return { x: sx / n, y: sy / n, v: sv / n };
        });
        return { t: frame.t, lm };
    });
}

// Fit the floor as a straight line y = a + b*x from the lowest foot points,
// which tolerates a slightly tilted camera.
function fitGround(frames, cfg) {
    const pts = [];
    for (const f of frames) {
        if (!f.lm) continue;
        for (const s of Object.values(SIDES)) {
            const he = f.lm[s.he], toe = f.lm[s.toe];
            if (Math.max(he.v, toe.v) < cfg.minVisibility) continue;
            pts.push({ x: he.x, y: Math.max(he.y, toe.y) });
        }
    }
    if (pts.length < 5) return null;
    const xs = pts.map(p => p.x);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const bins = 12, width = (maxX - minX) / bins || 1;
    const fit = [];
    for (let b = 0; b < bins; b++) {
        const inBin = pts.filter(p => p.x >= minX + b * width && p.x <= minX + (b + 1) * width);
        if (inBin.length < 5) continue;
        fit.push({ x: minX + (b + 0.5) * width, y: percentile(inBin.map(p => p.y), 90) });
    }
    if (!fit.length) return { a: percentile(pts.map(p => p.y), 90), b: 0 };
    if (fit.length === 1) return { a: fit[0].y, b: 0 };
    const n = fit.length;
    const mx = fit.reduce((s, p) => s + p.x, 0) / n;
    const my = fit.reduce((s, p) => s + p.y, 0) / n;
    let num = 0, den = 0;
    for (const p of fit) { num += (p.x - mx) * (p.y - my); den += (p.x - mx) ** 2; }
    const b = den ? num / den : 0;
    return { a: my - b * mx, b };
}

// ---------- main entry ----------

/**
 * @param {Array<{t:number, lm:Array<{x:number,y:number,v:number}>|null}>} rawFrames
 *        Per-frame landmarks in pixels, ordered by time (seconds).
 * @param {object} overrides Optional CONFIG overrides.
 */
export function analyzeApproach(rawFrames, overrides = {}) {
    const cfg = { ...CONFIG, ...overrides };
    const result = { ok: false, error: null, checks: [], events: {}, plants: [], series: [], metrics: {} };

    if (!rawFrames || rawFrames.length < 10) {
        result.error = "The clip is too short to analyze. Include the whole approach and the jump.";
        return result;
    }

    const covered = rawFrames.filter(f => f.lm).length / rawFrames.length;
    result.metrics.poseCoverage = round(covered);
    if (covered < cfg.minPoseCoverage) {
        result.error = `The body was only tracked in ${Math.round(covered * 100)}% of frames. Film from the side with the whole body in frame.`;
        return result;
    }

    const frames = smoothFrames(rawFrames, cfg.smoothWindow);
    const valid = frames.map((f, i) => (f.lm ? i : -1)).filter(i => i >= 0);
    const dt = median(frames.slice(1).map((f, i) => f.t - frames[i].t)) || 1 / 30;

    // Leg length (hip to knee to ankle) is the ruler for every distance.
    const legLen = median(valid.map(i => {
        const lm = frames[i].lm;
        const side = lm[L.an].v >= lm[R.an].v ? L : R;
        return dist(lm[side.hip], lm[side.kn]) + dist(lm[side.kn], lm[side.an]);
    }));
    if (!Number.isFinite(legLen) || legLen <= 0) {
        result.error = "Couldn't measure leg length. Make sure the hips, knees and ankles are visible.";
        return result;
    }
    result.metrics.legLengthPx = round(legLen, 1);

    const hip = frames.map(f => (f.lm ? mid(f.lm[L.hip], f.lm[R.hip]) : null));
    const sho = frames.map(f => (f.lm ? mid(f.lm[L.sh], f.lm[R.sh]) : null));

    const ground = fitGround(frames, cfg);
    if (!ground) {
        result.error = "Couldn't find the floor. Make sure the feet stay in frame.";
        return result;
    }
    const groundY = x => ground.a + ground.b * x;
    const hipH = hip.map(h => (h ? (groundY(h.x) - h.y) / legLen : NaN));

    // Apex: highest hip position in the clip.
    let apex = valid[0];
    for (const i of valid) if (hipH[i] > hipH[apex]) apex = i;

    const dir = Math.sign(hip[apex].x - hip[valid[0]].x) || 1;
    const travel = Math.abs(hip[apex].x - hip[valid[0]].x) / legLen;
    result.metrics.direction = dir > 0 ? "left-to-right" : "right-to-left";
    if (travel < 0.5) {
        result.error = "No approach movement found. Film from the side with the camera held still (don't follow the athlete), so the athlete runs across the frame.";
        return result;
    }

    // ----- foot contacts -----
    const contact = { left: [], right: [] };
    for (const [name, s] of Object.entries(SIDES)) {
        for (let i = 0; i < frames.length; i++) {
            const f = frames[i];
            if (!f.lm) { contact[name].push(false); continue; }
            const he = f.lm[s.he], toe = f.lm[s.toe];
            if (Math.max(he.v, toe.v) < cfg.minVisibility) { contact[name].push(false); continue; }
            const footY = Math.max(he.y, toe.y);
            const height = (groundY(he.x) - footY) / legLen;
            const prev = frames[i - 1] && frames[i - 1].lm, next = frames[i + 1] && frames[i + 1].lm;
            let speed = 0;
            if (prev && next) speed = Math.abs(next[s.he].x - prev[s.he].x) / (frames[i + 1].t - frames[i - 1].t) / legLen;
            contact[name].push(height < cfg.contactHeight && speed < cfg.contactSpeed);
        }
    }

    const anyContact = i => contact.left[i] || contact.right[i];
    let takeoffIdx = -1;
    for (let i = apex; i >= 0; i--) if (anyContact(i)) { takeoffIdx = i; break; }
    if (takeoffIdx < 0) {
        result.error = "Couldn't find the takeoff. Make sure the feet are visible on the floor before the jump.";
        return result;
    }
    const takeoffT = frames[takeoffIdx].t;
    const rise = hipH[apex] - hipH[takeoffIdx];
    result.metrics.hipRiseAfterTakeoff = round(rise);
    if (rise < cfg.minJumpRise) {
        result.error = "No jump found. Include the takeoff and the jump in the clip.";
        return result;
    }

    // Group contact frames into segments, one per foot plant.
    const segments = [];
    for (const [name, s] of Object.entries(SIDES)) {
        let start = -1;
        for (let i = 0; i <= frames.length; i++) {
            const on = i < frames.length && contact[name][i];
            if (on && start < 0) start = i;
            if (!on && start >= 0) {
                const end = i - 1;
                if (frames[end].t - frames[start].t + dt >= cfg.minContactSec) {
                    // Where the heel lands. The heel drifts forward as the foot
                    // rolls to the toes, so only use the first few frames.
                    const xs = [];
                    for (let k = start; k <= Math.min(end, start + 2); k++) xs.push(frames[k].lm[s.he].x);
                    segments.push({ side: name, i0: start, i1: end, t0: frames[start].t, t1: frames[end].t, x: median(xs) });
                }
                start = -1;
            }
        }
    }
    const firstT = frames[valid[0]].t;
    for (const s of segments) s.stance = s.t0 <= firstT + 2 * dt;
    segments.sort((a, b) => a.t0 - b.t0);

    // Plants before takeoff, merging contacts that are really the same plant:
    // the same foot touching again right away (heel-to-toe roll), or two
    // contacts at one spot (pose models sometimes swap left and right in side view).
    const plants = [];
    for (const s of segments) {
        if (s.stance || s.t0 > takeoffT) continue;
        const sameSpot = p => Math.abs(s.x - p.x) / legLen < cfg.mergePlantDistance;
        const sameFootAgain = p => p.side === s.side && s.t0 - p.t1 <= 0.15 && ((s.x - p.x) * dir) / legLen < 0.6;
        const last = [...plants].reverse().find(p => sameSpot(p) || sameFootAgain(p));
        const interveningOther = last && plants.some(p => p !== last && p.t0 > last.t0 && p.side !== last.side);
        if (last && !interveningOther && (sameSpot(last) ? s.t0 - last.t1 < 0.15 : true)) {
            last.t1 = Math.max(last.t1, s.t1);
            last.i1 = Math.max(last.i1, s.i1);
            continue;
        }
        plants.push({ ...s });
    }

    // Motion start: hips begin moving toward the net.
    let motionIdx = valid[0];
    for (let n = 1; n < valid.length - 3; n++) {
        const ok = [0, 1, 2].every(k => {
            const a = valid[n + k - 1], b = valid[n + k];
            return ((hip[b].x - hip[a].x) * dir) / (frames[b].t - frames[a].t) / legLen > cfg.motionStartSpeed;
        });
        if (ok) { motionIdx = valid[n - 1]; break; }
    }
    const motionT = frames[motionIdx].t;

    result.events = { motionStart: motionT, takeoff: takeoffT, apex: frames[apex].t };
    result.series = frames.map((f, i) => ({ t: f.t, hipH: round(hipH[i], 3) }));
    result.ground = ground;
    result.dir = dir;

    if (plants.length < 3) {
        result.error = `Found ${plants.length} step${plants.length === 1 ? "" : "s"} before takeoff. The clip needs the intent step, the penultimate step and the closing step.`;
        result.plants = plants;
        return result;
    }

    const n = plants.length;
    const intent = plants[n - 3], pen = plants[n - 2], close = plants[n - 1];
    intent.role = "intent"; pen.role = "penultimate"; close.role = "closing";
    result.plants = plants;
    Object.assign(result.events, { intent: intent.t0, penultimate: pen.t0, closing: close.t0 });

    // The foot already on the floor when a plant lands.
    const supportFor = plant => {
        const before = [...segments, ...plants]
            .filter(s => s.t0 < plant.t0 && s.t1 >= plant.t0 - 0.25 && Math.abs(s.x - plant.x) / legLen >= cfg.mergePlantDistance)
            .sort((a, b) => b.t0 - a.t0);
        return before[0] || null;
    };

    const stepLen = plant => {
        const sup = supportFor(plant);
        return sup ? ((plant.x - sup.x) * dir) / legLen : NaN;
    };

    const inWindow = (t0, t1) => valid.filter(i => frames[i].t >= t0 && frames[i].t <= t1);
    const add = check => result.checks.push(check);

    // ----- 1. Step lengths: medium, long, short -----
    const lens = { intent: stepLen(intent), penultimate: stepLen(pen), closing: stepLen(close) };
    result.metrics.stepLengths = { intent: round(lens.intent), penultimate: round(lens.penultimate), closing: round(lens.closing) };
    {
        const known = Object.values(lens).every(Number.isFinite);
        const penLongest = lens.penultimate > lens.intent && lens.penultimate > lens.closing;
        const closeShortest = lens.closing < lens.intent;
        add({
            id: "stepLengths",
            title: "Step lengths: medium, long, short",
            status: !known ? "na" : penLongest && closeShortest ? "pass" : penLongest ? "warn" : "fail",
            value: known
                ? `Intent ${lens.intent.toFixed(2)}, penultimate ${lens.penultimate.toFixed(2)}, closing ${lens.closing.toFixed(2)} leg lengths`
                : "Couldn't measure every step",
            cue: penLongest
                ? closeShortest ? "Penultimate is your longest step and the closing step is short." : "Shorten the closing step so the feet land close together."
                : "Make the penultimate your longest step. Reach with it and land heel first.",
        });
    }

    // ----- 2. Tempo: slow to fast -----
    {
        const prev = plants[n - 4];
        const T0 = intent.t0 - (prev ? prev.t0 : motionT);
        const T1 = pen.t0 - intent.t0;
        const T2 = close.t0 - pen.t0;
        const speedIn = (a, b) => {
            const idx = inWindow(a, b);
            if (idx.length < 2) return NaN;
            const i0 = idx[0], i1 = idx[idx.length - 1];
            return ((hip[i1].x - hip[i0].x) * dir) / (frames[i1].t - frames[i0].t) / legLen;
        };
        const vIntent = speedIn(prev ? prev.t0 : motionT, intent.t0);
        const vPen = speedIn(intent.t0, pen.t0);
        const timingOk = T1 <= T0 * cfg.tempoSlack && T2 <= T1 * cfg.tempoSlack;
        const speedOk = Number.isFinite(vIntent) && Number.isFinite(vPen) ? vPen >= vIntent : null;
        result.metrics.tempo = { stepTimes: [round(T0), round(T1), round(T2)], hipSpeedIntent: round(vIntent), hipSpeedPenultimate: round(vPen) };
        const score = (timingOk ? 1 : 0) + (speedOk ? 1 : 0);
        add({
            id: "tempo",
            title: "Tempo: slow to fast",
            status: score === 2 ? "pass" : score === 1 ? "warn" : "fail",
            value: `Step times ${T0.toFixed(2)}s, ${T1.toFixed(2)}s, ${T2.toFixed(2)}s` +
                (speedOk === null ? "" : `. Hip speed ${vIntent.toFixed(1)} then ${vPen.toFixed(1)} leg lengths/s`),
            cue: score === 2 ? "You speed up into the takeoff." : "Start slow and get faster each step. Don't cruise at one speed.",
        });
    }

    // ----- 3 + 4. Hip height: drop on the intent step, hold through the penultimate -----
    {
        const before = inWindow(-Infinity, intent.t0 - dt);
        const baseH = before.length >= 3 ? percentile(before.map(i => hipH[i]), 90) : NaN;
        const intentIdx = inWindow(intent.t0, pen.t0);
        const penIdx = inWindow(pen.t0, close.t0);
        const minOf = idx => (idx.length ? Math.min(...idx.map(i => hipH[i])) : NaN);
        const intentH = minOf(intentIdx);
        const penH = minOf(penIdx);
        const drop = baseH - intentH;
        const sink = intentH - penH;
        result.metrics.hipHeight = { start: round(baseH), intent: round(intentH), penultimate: round(penH), drop: round(drop), sink: round(sink) };

        add({
            id: "intentDrop",
            title: "Lower your hips on the intent step",
            status: !Number.isFinite(drop) ? "na" : drop >= cfg.intentHipDropMin ? "pass" : "warn",
            value: Number.isFinite(drop) ? `Hips dropped ${(drop * 100).toFixed(0)}% of leg length` : "Start the clip a step earlier to measure this",
            cue: drop >= cfg.intentHipDropMin ? "You get low early, on the intent step." : "Drop your hips on the intent step, the step before your penultimate.",
        });

        const sinkStatus = !Number.isFinite(sink) ? "na" : sink <= cfg.penultimateSinkMax ? "pass" : sink <= cfg.penultimateSinkMax * 2 ? "warn" : "fail";
        add({
            id: "penultimateHold",
            title: "Hold hip height through the penultimate",
            status: sinkStatus,
            value: Number.isFinite(sink)
                ? sink > 0 ? `Hips sank another ${(sink * 100).toFixed(0)}% of leg length on the penultimate` : "Hips stayed level on the penultimate"
                : "Couldn't measure",
            cue: sinkStatus === "pass"
                ? "You carry your horizontal momentum into takeoff."
                : "Don't sink on the penultimate. Stay at the height you set on the intent step.",
        });
    }

    // ----- 5. Starting lean -----
    const leanAt = i => (Math.atan2(dir * (sho[i].x - hip[i].x), hip[i].y - sho[i].y) * 180) / Math.PI;
    {
        const startIdx = valid.filter(i => frames[i].t <= motionT);
        const lean = startIdx.length >= 3 ? median(startIdx.map(leanAt)) : NaN;
        result.metrics.startLean = round(lean, 1);
        let status = "na";
        if (Number.isFinite(lean)) {
            if (lean >= cfg.startLeanMin && lean <= cfg.startLeanMax) status = "pass";
            else if (lean >= cfg.startLeanMin - cfg.startLeanSlack && lean <= cfg.startLeanMax + cfg.startLeanSlack) status = "warn";
            else status = "fail";
        }
        add({
            id: "startLean",
            title: "Starting lean: 20-30 degrees forward",
            status,
            value: Number.isFinite(lean) ? `${lean.toFixed(0)} degrees` : "Start the clip before the first step to measure this",
            cue: status === "pass" ? "Good ready position." : "Start with a 20-30 degree forward lean, weight on the front foot.",
        });
    }

    // ----- 6. Chest position through the approach and plant -----
    {
        const approachLean = median(inWindow(intent.t0, close.t0).map(leanAt));
        const plantLean = leanAt(takeoffIdx);
        result.metrics.approachLean = round(approachLean, 1);
        result.metrics.plantLean = round(plantLean, 1);
        const forward = approachLean >= cfg.approachLeanMin;
        const slight = plantLean <= cfg.plantLeanMax && plantLean >= -5;
        add({
            id: "chest",
            title: "Chest ahead of hips, slight lean at the plant",
            status: forward && slight ? "pass" : forward || slight ? "warn" : "fail",
            value: `Approach ${approachLean.toFixed(0)} degrees, plant ${plantLean.toFixed(0)} degrees`,
            cue: !forward ? "Keep your chest ahead of your hips as you approach." : !slight ? "Only a slight lean at the plant, so you go up, not out." : "Good chest position.",
        });
    }

    // ----- 7-10. Arm swing -----
    {
        const winIdx = inWindow(intent.t0, frames[apex].t);
        const visOf = s => median(winIdx.map(i => frames[i].lm[s.wr].v));
        const near = visOf(L) >= visOf(R) ? L : R;
        const far = near === L ? R : L;
        const rel = (i, s) => ((frames[i].lm[s.wr].x - frames[i].lm[s.sh].x) * dir) / legLen;

        const swingIdx = inWindow(intent.t0, takeoffT);
        let peak = swingIdx[0];
        for (const i of swingIdx) if (rel(i, near) < rel(peak, near)) peak = i;
        const peakRel = rel(peak, near);
        const peakT = frames[peak].t;
        result.metrics.arms = { nearArm: near === L ? "left" : "right", backswingPeakT: round(peakT, 3), wristBehindShoulder: round(-peakRel) };

        const back = peakRel <= -cfg.armBackMin;
        const timed = peakT >= pen.t0 - cfg.armBackEarlySec;
        add({
            id: "armBack",
            title: "Arms swing back on the last two steps",
            status: back && timed ? "pass" : back ? "warn" : "fail",
            value: back
                ? `Full backswing at ${peakT.toFixed(2)}s, penultimate plant at ${pen.t0.toFixed(2)}s`
                : "Wrist never got behind the shoulder",
            cue: back && timed ? "Arms go back as you plant." : back ? "Your arms go back too early. Time the backswing with the penultimate and closing steps." : "Swing both arms all the way back during the last two steps.",
        });

        const upIdx = inWindow(takeoffT, frames[apex].t);
        const up = upIdx.some(i => frames[i].lm[near.wr].y < frames[i].lm[near.sh].y);
        add({
            id: "armUp",
            title: "Arms drive up at takeoff",
            status: upIdx.length ? (up ? "pass" : "fail") : "na",
            value: up ? "Wrist above shoulder after takeoff" : "Wrist stayed below shoulder",
            cue: up ? "Forward, back, up." : "Drive both arms up hard as you leave the floor.",
        });

        const elbow = angleDeg(frames[peak].lm[near.sh], frames[peak].lm[near.el], frames[peak].lm[near.wr]);
        result.metrics.arms.elbowAtBackswing = round(elbow, 0);
        add({
            id: "chickenWing",
            title: "Straight arm swing, no chicken wing",
            status: !Number.isFinite(elbow) ? "na" : elbow >= cfg.straightElbowMin ? "pass" : elbow >= cfg.straightElbowWarn ? "warn" : "fail",
            value: Number.isFinite(elbow) ? `Elbow ${elbow.toFixed(0)} degrees at full backswing` : "Couldn't see the elbow",
            cue: elbow >= cfg.straightElbowMin ? "Long arms straight back." : "Keep the arms long and swing them straight back instead of bending and flaring the elbows.",
        });

        const both = inWindow(pen.t0 - 0.2, takeoffT).filter(i =>
            frames[i].lm[near.wr].v >= cfg.minVisibility && frames[i].lm[far.wr].v >= cfg.farArmVisibility);
        let status = "na", value = "Far arm hidden from this angle";
        if (both.length >= 5) {
            const scissored = both.filter(i => {
                const a = rel(i, near), b = rel(i, far);
                return Math.sign(a) !== Math.sign(b) && Math.abs(a - b) >= cfg.scissorGap;
            }).length / both.length;
            result.metrics.arms.scissoredShare = round(scissored);
            status = scissored <= cfg.scissorWarn ? "pass" : scissored <= cfg.scissorFail ? "warn" : "fail";
            value = `Arms out of sync in ${(scissored * 100).toFixed(0)}% of frames`;
        }
        add({
            id: "scissor",
            title: "Arms swing together, no scissored arms",
            status,
            value,
            cue: status === "pass" ? "Both arms move as a pair." : status === "na" ? "Side view hides the far arm, so this check needs a front or 45-degree clip." : "Swing both arms back and up together, not one forward and one back.",
        });
    }

    result.ok = true;
    return result;
}
