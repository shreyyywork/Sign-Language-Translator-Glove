/**
 * Interactive Hand Skeletal Visualizer for ISL Glove
 * Styled for GitHub Dark theme: #010409 canvas, #161b22 palm, #30363d borders,
 * GitHub accent colored finger flexion bones.
 */

class HandVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        
        this.flexValues = { thumb: 0.0, index: 0.0, middle: 0.0, ring: 0.0, little: 0.0 };
        this.imu = { acc_x: 0.0, acc_y: 0.98, acc_z: 0.0, gyro_x: 0.0, gyro_y: 0.0, gyro_z: 0.0 };
        
        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.render();
    }

    resize() {
        if (!this.canvas) return;
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * window.devicePixelRatio || 500;
        this.canvas.height = rect.height * window.devicePixelRatio || 300;
    }

    update(flexData, imuData) {
        if (flexData) {
            this.flexValues = { ...this.flexValues, ...flexData };
        }
        if (imuData) {
            this.imu = { ...this.imu, ...imuData };
        }
    }

    render() {
        requestAnimationFrame(() => this.render());
        if (!this.ctx || !this.canvas) return;

        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        ctx.clearRect(0, 0, w, h);

        // GitHub subtle canvas grid
        this.drawBackgroundGrid(ctx, w, h);

        ctx.save();
        // Move to palm root origin
        const originX = w * 0.5;
        const originY = h * 0.72;
        ctx.translate(originX, originY);

        // Apply IMU Tilt (Pitch from acc_x, Roll from acc_z)
        const pitch = Math.atan2(this.imu.acc_x || 0, this.imu.acc_y || 1) * 0.6;
        const roll = Math.atan2(this.imu.acc_z || 0, this.imu.acc_y || 1) * 0.6;
        ctx.rotate(pitch);
        ctx.scale(1.0, Math.cos(roll) * 0.4 + 0.6);

        // Draw Palm Plate (GitHub dark box style)
        this.drawPalm(ctx);

        // Draw 5 Fingers
        const fingers = [
            { name: 'thumb',  rootX: -45, rootY: -15, angle: -50, len: 26, joints: 3, flex: this.flexValues.thumb },
            { name: 'index',  rootX: -32, rootY: -70, angle: -15, len: 32, joints: 3, flex: this.flexValues.index },
            { name: 'middle', rootX: -2,  rootY: -78, angle: 0,   len: 36, joints: 3, flex: this.flexValues.middle },
            { name: 'ring',   rootX: 28,  rootY: -72, angle: 12,  len: 33, joints: 3, flex: this.flexValues.ring },
            { name: 'little', rootX: 52,  rootY: -55, angle: 26,  len: 26, joints: 3, flex: this.flexValues.little }
        ];

        fingers.forEach(f => this.drawFinger(ctx, f));

        // Center IMU crosshair indicator
        this.drawIMUIndicator(ctx);

        ctx.restore();
    }

    drawBackgroundGrid(ctx, w, h) {
        ctx.strokeStyle = 'rgba(48, 54, 61, 0.4)'; // #30363d
        ctx.lineWidth = 1;
        const step = 28;
        for (let x = 0; x < w; x += step) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }
        for (let y = 0; y < h; y += step) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }
    }

    drawPalm(ctx) {
        ctx.save();
        ctx.fillStyle = '#161b22';
        ctx.strokeStyle = '#30363d';
        ctx.lineWidth = 2;

        ctx.beginPath();
        ctx.moveTo(-55, 10);
        ctx.lineTo(-50, -40);
        ctx.lineTo(-35, -70);
        ctx.lineTo(35, -70);
        ctx.lineTo(58, -45);
        ctx.lineTo(50, 10);
        ctx.closePath();

        ctx.fill();
        ctx.stroke();

        // Palm center circuit lines
        ctx.strokeStyle = '#58a6ff';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(0, -25, 15, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
    }

    drawFinger(ctx, finger) {
        ctx.save();
        ctx.translate(finger.rootX, finger.rootY);
        ctx.rotate((finger.angle * Math.PI) / 180);

        let curX = 0;
        let curY = 0;
        const flex = Math.min(Math.max(finger.flex || 0, 0), 1);
        const bendPerJoint = flex * 0.48;

        const boneColor = this.getStressColor(flex);

        ctx.strokeStyle = boneColor;
        ctx.lineWidth = 4.5;
        ctx.lineCap = 'round';

        for (let j = 0; j < finger.joints; j++) {
            ctx.rotate(bendPerJoint);
            const nextX = curX;
            const nextY = curY - finger.len * (1.0 - flex * 0.15);

            // Draw Bone
            ctx.beginPath();
            ctx.moveTo(curX, curY);
            ctx.lineTo(nextX, nextY);
            ctx.stroke();

            // Draw Joint Node
            ctx.fillStyle = '#e6edf3';
            ctx.beginPath();
            ctx.arc(nextX, nextY, 3, 0, Math.PI * 2);
            ctx.fill();

            curX = nextX;
            curY = nextY;
        }

        // Draw Fingertip Sensor
        ctx.fillStyle = boneColor;
        ctx.beginPath();
        ctx.arc(curX, curY, 4, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
    }

    getStressColor(flex) {
        if (flex < 0.25) return '#58a6ff';      // GitHub Blue
        if (flex < 0.55) return '#3fb950';      // GitHub Green
        if (flex < 0.80) return '#d29922';      // GitHub Orange / Attention
        return '#f85149';                       // GitHub Red / Danger
    }

    drawIMUIndicator(ctx) {
        ctx.save();
        ctx.translate(0, -25);
        ctx.strokeStyle = '#8b949e';
        ctx.lineWidth = 1;

        // Draw crosshairs
        ctx.beginPath();
        ctx.moveTo(-8, 0); ctx.lineTo(8, 0);
        ctx.moveTo(0, -8); ctx.lineTo(0, 8);
        ctx.stroke();

        // Accel dot
        const dotX = (this.imu.acc_x || 0) * 10;
        const dotY = (this.imu.acc_z || 0) * 10;
        ctx.fillStyle = '#58a6ff';
        ctx.beginPath();
        ctx.arc(dotX, dotY, 3, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
    }
}

window.HandVisualizer = HandVisualizer;
