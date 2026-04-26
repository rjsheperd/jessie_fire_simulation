// Playwright test: verify WebGPU fire simulation runs in Brave browser
import { chromium } from 'playwright';

const SERVER = 'http://localhost:5001';

(async () => {
    console.log('Launching Brave browser...');
    const browser = await chromium.launch({
        executablePath: '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
        headless: false,
        args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan,UseSkiaRenderer',
            '--start-fullscreen',
        ],
    });

    const context = await browser.newContext({
        viewport: null,
    });
    const page = await context.newPage();

    // Collect console messages
    const consoleLogs = [];
    page.on('console', msg => {
        const text = msg.text();
        consoleLogs.push(text);
        console.log(`[browser ${msg.type()}] ${text}`);
    });
    page.on('pageerror', err => {
        console.log(`[page error] ${err.message}`);
    });

    try {
        // Load the fire visualization page
        console.log('Loading fire_vis page...');
        await page.goto(`${SERVER}/fire_vis/unr_aa_1`, { waitUntil: 'domcontentloaded', timeout: 30000 });

        // Wait for WebGPU simulation to finish
        console.log('Waiting for WebGPU simulation to complete...');
        await page.waitForFunction(() => window.webgpuSimEnabled === true, { timeout: 120000 });
        console.log('WebGPU simulation done.');

        // Move canvas out of container, make it fullscreen, hide everything else
        await page.evaluate(() => {
            const canvas = document.getElementById('myCanvas');

            // Move canvas to body
            document.body.appendChild(canvas);

            // Hide the container (everything else)
            document.querySelector('.container').style.display = 'none';

            // Fullscreen styling
            document.body.style.margin = '0';
            document.body.style.padding = '0';
            document.body.style.overflow = 'hidden';
            document.body.style.background = '#000';

            canvas.style.border = 'none';
            canvas.style.display = 'block';
            canvas.style.position = 'fixed';
            canvas.style.top = '0';
            canvas.style.left = '0';
            canvas.style.width = '100vw';
            canvas.style.height = '100vh';
            canvas.style.objectFit = 'contain';
            canvas.style.imageRendering = 'pixelated';
        });

        // F11 fullscreen
        await page.keyboard.press('F11');
        await page.waitForTimeout(500);

        // Click Start/Continue to begin fire animation
        console.log('Starting fire simulation animation...');
        await page.evaluate(() => {
            document.getElementById('startButtonID').click();
        });

        // Take periodic screenshots as fire spreads
        console.log('Fire simulation running. Taking screenshots...');
        for (let i = 1; i <= 5; i++) {
            await page.waitForTimeout(5000);
            await page.screenshot({ path: `test_screenshot_${i}.png` });
            console.log(`Screenshot ${i}/5 saved (t=${i * 5}s)`);
        }

        await page.screenshot({ path: 'test_screenshot.png' });
        console.log('Final screenshot saved.');

        const webgpuLogs = consoleLogs.filter(l => l.includes('WebGPU'));
        console.log('\nWebGPU log summary:');
        webgpuLogs.forEach(l => console.log('  ', l));
        console.log('\nPASS: WebGPU simulation completed and animation started.');

    } catch (err) {
        console.error('Test failed:', err.message);
        await page.screenshot({ path: 'test_error_screenshot.png' });
    } finally {
        console.log('Keeping browser open for 15 seconds...');
        await page.waitForTimeout(15000);
        await browser.close();
        console.log('Done.');
    }
})();
