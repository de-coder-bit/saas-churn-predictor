import { useMemo } from "react";

// Radial instrument gauge: 0 -> 1 probability mapped across a 220deg arc.
// Signature element of the app: styled like a pressure/fuel gauge.
export default function Gauge({ probability = 0, size = 280 }) {
  const startAngle = -200; // degrees
  const endAngle = 20;
  const sweep = endAngle - startAngle;

  const clamped = Math.min(Math.max(probability, 0), 1);
  const needleAngle = startAngle + clamped * sweep;

  const cx = size / 2;
  const cy = size / 2 + 10;
  const r = size / 2 - 28;

  const toXY = (angleDeg, radius) => {
    const rad = (angleDeg * Math.PI) / 180;
    return [cx + radius * Math.cos(rad), cy + radius * Math.sin(rad)];
  };

  const ticks = useMemo(() => {
    const arr = [];
    const majorCount = 10;
    for (let i = 0; i <= majorCount; i++) {
      const t = i / majorCount;
      const angle = startAngle + t * sweep;
      const [x1, y1] = toXY(angle, r);
      const [x2, y2] = toXY(angle, r - 12);
      arr.push({ x1, y1, x2, y2, major: true });
    }
    const minorCount = 50;
    for (let i = 0; i <= minorCount; i++) {
      if (i % 5 === 0) continue;
      const t = i / minorCount;
      const angle = startAngle + t * sweep;
      const [x1, y1] = toXY(angle, r);
      const [x2, y2] = toXY(angle, r - 6);
      arr.push({ x1, y1, x2, y2, major: false });
    }
    return arr;
  }, [size]);

  const arcPath = (a0, a1, radius) => {
    const [x1, y1] = toXY(a0, radius);
    const [x2, y2] = toXY(a1, radius);
    const largeArc = a1 - a0 > 180 ? 1 : 0;
    return `M ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2}`;
  };

  const [needleX, needleY] = toXY(needleAngle, r - 18);

  const bandColor = clamped < 0.15 ? "var(--teal-signal)" : clamped < 0.35 ? "#C9C15A" : clamped < 0.6 ? "var(--amber)" : "var(--coral)";

  return (
    <svg width={size} height={size * 0.72} viewBox={`0 0 ${size} ${size * 0.72}`} role="img" aria-label={`Churn probability gauge: ${Math.round(clamped * 100)} percent`}>
      {/* base track */}
      <path d={arcPath(startAngle, endAngle, r)} fill="none" stroke="var(--hairline)" strokeWidth="10" strokeLinecap="round" />
      {/* colored progress band */}
      <path
        d={arcPath(startAngle, needleAngle, r)}
        fill="none"
        stroke={bandColor}
        strokeWidth="10"
        strokeLinecap="round"
        style={{ transition: "d 0.6s cubic-bezier(.4,0,.2,1)" }}
      />
      {/* ticks */}
      {ticks.map((t, i) => (
        <line
          key={i}
          x1={t.x1} y1={t.y1} x2={t.x2} y2={t.y2}
          stroke="var(--paper-dim)"
          strokeWidth={t.major ? 2 : 1}
          opacity={t.major ? 0.8 : 0.35}
        />
      ))}
      {/* needle */}
      <g style={{ transition: "transform 0.6s cubic-bezier(.4,0,.2,1)" }}>
        <line x1={cx} y1={cy} x2={needleX} y2={needleY} stroke="var(--paper)" strokeWidth="3" strokeLinecap="round" />
        <circle cx={cx} cy={cy} r="7" fill="var(--paper)" />
        <circle cx={cx} cy={cy} r="3" fill={bandColor} />
      </g>
      {/* readout */}
      <text x={cx} y={cy - 6} textAnchor="middle" fontFamily="var(--font-mono)" fontSize="34" fontWeight="600" fill="var(--paper)">
        {Math.round(clamped * 100)}%
      </text>
      <text x={cx} y={cy + 18} textAnchor="middle" fontFamily="var(--font-mono)" fontSize="11" letterSpacing="2" fill="var(--paper-dim)">
        CHURN RISK
      </text>
    </svg>
  );
}
