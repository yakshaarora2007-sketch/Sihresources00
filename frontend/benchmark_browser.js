import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const CHROME_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';

async function runBenchmark() {
  console.log('Launching headless Chrome for live React performance instrumentation...');
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu-vsync'],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900 });

    console.log('Navigating to http://localhost:5173...');
    await page.goto('http://localhost:5173', { waitUntil: 'networkidle0' });

    // Wait for canvas and initial frame to load
    await page.waitForSelector('canvas', { timeout: 10000 });
    console.log('Canvas loaded. Waiting for frame 0 telemetry...');
    await page.waitForFunction(() => window.__LIDAR_TELEMETRY__ !== undefined, { timeout: 10000 });

    // Start playback via Space key
    console.log('Triggering playback (Space)...');
    await page.keyboard.press('Space');

    // Run observation for 10 seconds, sampling every 100ms
    console.log('Sampling live browser performance metrics over 10.0 seconds...');
    const samples = [];
    const startTime = Date.now();
    const durationMs = 10000;

    while (Date.now() - startTime < durationMs) {
      await new Promise(r => setTimeout(r, 100));
      const telemetry = await page.evaluate(() => {
        return window.__LIDAR_TELEMETRY__ || null;
      });
      if (telemetry) {
        samples.push(telemetry);
      }
    }

    // Stop playback
    await page.keyboard.press('Space');

    if (samples.length === 0) {
      throw new Error('No telemetry samples collected!');
    }

    console.log(`Collected ${samples.length} browser telemetry samples.`);

    // Compute statistics
    const stats = (arr) => {
      const min = Math.min(...arr);
      const max = Math.max(...arr);
      const avg = arr.reduce((a, b) => a + b, 0) / arr.length;
      return { min, avg, max };
    };

    const fetchStats = stats(samples.map(s => s.fetchTimeMs));
    const parseStats = stats(samples.map(s => s.parseTimeMs));
    const updateStats = stats(samples.map(s => s.updateTimeMs));
    const renderStats = stats(samples.map(s => s.renderTimeMs));
    const totalStats = stats(samples.map(s => s.totalFrameTimeMs));
    const fpsStats = stats(samples.map(s => s.actualFps));

    const finalSample = samples[samples.length - 1];

    const benchmarkReport = {
      timestamp: new Date().toISOString(),
      observationWindowSec: durationMs / 1000,
      totalSamples: samples.length,
      targetFps: 10.0,
      frameBudgetMs: 100.0,
      measurements: {
        dataFetchMs: fetchStats,
        dataParseMs: parseStats,
        frameUpdateMs: updateStats,
        canvasRenderMs: renderStats,
        endToEndFrameMs: totalStats,
        fps: fpsStats,
      },
      telemetrySnapshot: {
        actualFps: finalSample.actualFps,
        avgFps: finalSample.avgFps,
        maxFrameTimeMs: finalSample.maxFrameTimeMs,
        droppedFrames: finalSample.droppedFrames,
        cacheHits: finalSample.cacheHits,
        cacheMisses: finalSample.cacheMisses,
        cachedFramesCount: finalSample.cachedFramesCount,
      }
    };

    console.log('\n==================================================');
    console.log('REACT FRONTEND LIVE BROWSER PERFORMANCE REPORT');
    console.log('==================================================');
    console.log(`Observation window: ${durationMs / 1000} s (${samples.length} samples)`);
    console.log(`Actual FPS: average ${fpsStats.avg.toFixed(2)} FPS (min ${fpsStats.min.toFixed(2)}, max ${fpsStats.max.toFixed(2)})`);
    console.log(`Data fetch/cache-hit time: average ${fetchStats.avg.toFixed(2)} ms (min ${fetchStats.min.toFixed(2)}, max ${fetchStats.max.toFixed(2)})`);
    console.log(`Data decode/parse time: average ${parseStats.avg.toFixed(3)} ms (min ${parseStats.min.toFixed(3)}, max ${parseStats.max.toFixed(3)})`);
    console.log(`Frame update time: average ${updateStats.avg.toFixed(2)} ms (min ${updateStats.min.toFixed(2)}, max ${updateStats.max.toFixed(2)})`);
    console.log(`Canvas 2D render time: average ${renderStats.avg.toFixed(2)} ms (min ${renderStats.min.toFixed(2)}, max ${renderStats.max.toFixed(2)})`);
    console.log(`End-to-end frame latency: average ${totalStats.avg.toFixed(2)} ms (min ${totalStats.min.toFixed(2)}, max ${totalStats.max.toFixed(2)})`);
    console.log(`100ms Frame Budget Utilization: ${(totalStats.avg / 100 * 100).toFixed(1)}%`);
    console.log(`Dropped frames count: ${finalSample.droppedFrames}`);
    console.log(`Cache hits: ${finalSample.cacheHits}, misses: ${finalSample.cacheMisses}`);
    console.log(`Bounded cache memory footprint: ${finalSample.cachedFramesCount} / 15 frames`);
    console.log('==================================================\n');

    fs.writeFileSync('benchmark_results_react.json', JSON.stringify(benchmarkReport, null, 2));
    console.log('Saved benchmark results to benchmark_results_react.json');

  } finally {
    await browser.close();
  }
}

runBenchmark().catch(err => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});
