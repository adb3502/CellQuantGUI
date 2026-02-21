<script lang="ts">
	/**
	 * A circular gauge showing biological age vs chronological age.
	 * Renders as an SVG dial with an animated needle.
	 */

	let {
		chronologicalAge = 50,
		biologicalAge = 50,
		confidence = 0.85
	}: {
		chronologicalAge: number;
		biologicalAge: number;
		confidence: number;
	} = $props();

	const SIZE = 260;
	const CX = SIZE / 2;
	const CY = SIZE / 2 + 10;
	const RADIUS = 100;
	const START_ANGLE = -210;
	const END_ANGLE = 30;
	const RANGE = END_ANGLE - START_ANGLE;

	// Map age to angle (20-100 range)
	function ageToAngle(age: number): number {
		const clamped = Math.max(20, Math.min(100, age));
		const frac = (clamped - 20) / 80;
		return START_ANGLE + frac * RANGE;
	}

	function polarToCart(angle: number, r: number): { x: number; y: number } {
		const rad = (angle * Math.PI) / 180;
		return { x: CX + r * Math.cos(rad), y: CY + r * Math.sin(rad) };
	}

	function describeArc(startAngle: number, endAngle: number, r: number): string {
		const s = polarToCart(startAngle, r);
		const e = polarToCart(endAngle, r);
		const largeArc = endAngle - startAngle > 180 ? 1 : 0;
		return `M ${s.x} ${s.y} A ${r} ${r} 0 ${largeArc} 1 ${e.x} ${e.y}`;
	}

	let bioAngle = $derived(ageToAngle(biologicalAge));
	let chronoAngle = $derived(ageToAngle(chronologicalAge));
	let ageGap = $derived(biologicalAge - chronologicalAge);
	let gapColor = $derived(ageGap > 2 ? 'var(--error)' : ageGap < -2 ? 'var(--success)' : 'var(--accent)');

	// Tick marks
	const ticks = Array.from({ length: 9 }, (_, i) => {
		const age = 20 + i * 10;
		const angle = ageToAngle(age);
		const inner = polarToCart(angle, RADIUS - 8);
		const outer = polarToCart(angle, RADIUS + 2);
		const label = polarToCart(angle, RADIUS + 16);
		return { age, inner, outer, label };
	});

	// Needle endpoint
	let needleTip = $derived(polarToCart(bioAngle, RADIUS - 18));
	let chronoMark = $derived(polarToCart(chronoAngle, RADIUS - 4));
</script>

<div class="dial-container">
	<svg width={SIZE} height={SIZE} viewBox="0 0 {SIZE} {SIZE}">
		<!-- Background arc -->
		<path
			d={describeArc(START_ANGLE, END_ANGLE, RADIUS)}
			fill="none"
			stroke="var(--border)"
			stroke-width="12"
			stroke-linecap="round"
		/>

		<!-- Colored zone: green (young) → yellow → red (old) -->
		<path
			d={describeArc(START_ANGLE, START_ANGLE + RANGE * 0.35, RADIUS)}
			fill="none"
			stroke="var(--success)"
			stroke-width="12"
			stroke-linecap="round"
			opacity="0.3"
		/>
		<path
			d={describeArc(START_ANGLE + RANGE * 0.35, START_ANGLE + RANGE * 0.65, RADIUS)}
			fill="none"
			stroke="var(--warning)"
			stroke-width="12"
			opacity="0.3"
		/>
		<path
			d={describeArc(START_ANGLE + RANGE * 0.65, END_ANGLE, RADIUS)}
			fill="none"
			stroke="var(--error)"
			stroke-width="12"
			stroke-linecap="round"
			opacity="0.3"
		/>

		<!-- Tick marks -->
		{#each ticks as tick}
			<line
				x1={tick.inner.x}
				y1={tick.inner.y}
				x2={tick.outer.x}
				y2={tick.outer.y}
				stroke="var(--text-muted)"
				stroke-width="1.5"
			/>
			<text
				x={tick.label.x}
				y={tick.label.y}
				text-anchor="middle"
				dominant-baseline="middle"
				fill="var(--text-faint)"
				font-size="9"
				font-family="var(--font-mono)"
			>
				{tick.age}
			</text>
		{/each}

		<!-- Chronological age marker -->
		<circle
			cx={chronoMark.x}
			cy={chronoMark.y}
			r="4"
			fill="var(--text-muted)"
			stroke="var(--bg-elevated)"
			stroke-width="1.5"
		/>

		<!-- Needle -->
		<line
			x1={CX}
			y1={CY}
			x2={needleTip.x}
			y2={needleTip.y}
			stroke={gapColor}
			stroke-width="2.5"
			stroke-linecap="round"
		/>
		<circle cx={CX} cy={CY} r="5" fill={gapColor} />

		<!-- Center text -->
		<text
			x={CX}
			y={CY + 32}
			text-anchor="middle"
			fill={gapColor}
			font-size="28"
			font-weight="600"
			font-family="var(--font-mono)"
		>
			{biologicalAge.toFixed(1)}
		</text>
		<text
			x={CX}
			y={CY + 48}
			text-anchor="middle"
			fill="var(--text-muted)"
			font-size="10"
			font-family="var(--font-ui)"
		>
			Biological Age
		</text>
	</svg>

	<div class="dial-stats">
		<div class="dial-stat">
			<span class="dial-stat-label font-ui">Chrono</span>
			<span class="dial-stat-value font-mono">{chronologicalAge}</span>
		</div>
		<div class="dial-stat">
			<span class="dial-stat-label font-ui">Gap</span>
			<span class="dial-stat-value font-mono" style="color: {gapColor}">
				{ageGap > 0 ? '+' : ''}{ageGap.toFixed(1)}y
			</span>
		</div>
		<div class="dial-stat">
			<span class="dial-stat-label font-ui">Confidence</span>
			<span class="dial-stat-value font-mono">{(confidence * 100).toFixed(0)}%</span>
		</div>
	</div>
</div>

<style>
	.dial-container {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 8px;
	}

	.dial-stats {
		display: flex;
		gap: 24px;
	}

	.dial-stat {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 2px;
	}

	.dial-stat-label {
		font-size: 10px;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	.dial-stat-value {
		font-size: 14px;
		font-weight: 500;
	}
</style>
