/* ═══════════════════════════════════════════════════════════════
   ASRIS — Frontend Application
   Backend integration · Dynamic interactions · Particle system
   ═══════════════════════════════════════════════════════════════ */

const API_BASE = window.location.origin;

// ─── Navigation ─────────────────────────────────────────────────

document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const section = link.dataset.section;
        switchSection(section);
    });
});

function switchSection(sectionName) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));

    const section = document.getElementById(`section-${sectionName}`);
    const link = document.querySelector(`[data-section="${sectionName}"]`);

    if (section) section.classList.add('active');
    if (link) link.classList.add('active');

    // Load dashboard data on switch
    if (sectionName === 'dashboard') loadDashboard();
}

// ─── Server Status ──────────────────────────────────────────────

async function checkServerStatus() {
    const dot = document.querySelector('.status-dot');
    const text = document.querySelector('.status-text');

    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        if (data.status === 'healthy') {
            dot.className = 'status-dot online';
            text.textContent = `Online · v${data.version}`;
        }
    } catch {
        dot.className = 'status-dot offline';
        text.textContent = 'Offline';
    }
}

checkServerStatus();
setInterval(checkServerStatus, 15000);

// ─── Particles ──────────────────────────────────────────────────

function createParticles() {
    const container = document.getElementById('particles');
    const count = 25;

    for (let i = 0; i < count; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';

        const size = Math.random() * 3 + 1;
        const left = Math.random() * 100;
        const duration = Math.random() * 15 + 10;
        const delay = Math.random() * 15;
        const hue = Math.random() > 0.5 ? '239' : '270';

        particle.style.cssText = `
            width: ${size}px;
            height: ${size}px;
            left: ${left}%;
            background: hsla(${hue}, 80%, 70%, ${Math.random() * 0.4 + 0.1});
            animation-duration: ${duration}s;
            animation-delay: -${delay}s;
        `;

        container.appendChild(particle);
    }
}

createParticles();

// ─── PDF Upload System ──────────────────────────────────────────

const uploadedResumes = {}; // {filename: extracted_text}

// Toggle text input visibility
document.getElementById('toggleTextInput').addEventListener('click', () => {
    const inputs = document.getElementById('resumeInputs');
    const btn = document.getElementById('toggleTextInput');
    if (inputs.style.display === 'none') {
        inputs.style.display = 'flex';
        btn.textContent = 'Hide text input ▴';
    } else {
        inputs.style.display = 'none';
        btn.textContent = 'Or paste text manually ▾';
    }
});

// ─── Rank Section Drop Zone ─────────────────────────────────────

function setupDropZone(dropZoneId, fileInputId, browseBtnId, onFiles) {
    const zone = document.getElementById(dropZoneId);
    const input = document.getElementById(fileInputId);
    const browseBtn = document.getElementById(browseBtnId);

    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        input.click();
    });

    zone.addEventListener('click', () => input.click());

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('drag-over');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        const files = Array.from(e.dataTransfer.files).filter(f => f.name.toLowerCase().endsWith('.pdf'));
        if (files.length > 0) onFiles(files);
        else showToast('Please drop PDF files only.', 'error');
    });

    input.addEventListener('change', () => {
        const files = Array.from(input.files);
        if (files.length > 0) onFiles(files);
        input.value = '';
    });
}

// Upload PDFs for ranking
setupDropZone('rankDropZone', 'rankFileInput', 'rankBrowseBtn', async (files) => {
    const container = document.getElementById('uploadedFiles');

    for (const file of files) {
        // Show loading badge
        const id = `file-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`;
        const badge = document.createElement('div');
        badge.className = 'uploaded-file';
        badge.id = id;
        badge.innerHTML = `
            <div class="uploaded-file-icon">📄</div>
            <div class="uploaded-file-info">
                <div class="uploaded-file-name">${escapeHtml(file.name)}</div>
                <div class="uploaded-file-meta"><span class="upload-spinner"></span> Extracting text...</div>
            </div>
        `;
        container.appendChild(badge);

        try {
            const formData = new FormData();
            formData.append('file', file);

            const res = await fetch(`${API_BASE}/upload-pdf`, { method: 'POST', body: formData });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Upload failed');
            }

            const data = await res.json();
            uploadedResumes[data.filename] = data.text;

            // Update badge to success
            badge.innerHTML = `
                <div class="uploaded-file-icon">✅</div>
                <div class="uploaded-file-info">
                    <div class="uploaded-file-name">${escapeHtml(data.filename)}</div>
                    <div class="uploaded-file-meta">${data.word_count} words extracted</div>
                </div>
                <button class="btn-remove" title="Remove" data-file="${escapeHtml(data.filename)}">&times;</button>
            `;

            badge.querySelector('.btn-remove').addEventListener('click', () => {
                delete uploadedResumes[data.filename];
                badge.style.animation = 'fadeInUp 0.3s ease reverse';
                setTimeout(() => badge.remove(), 250);
            });

            showToast(`Extracted ${data.word_count} words from ${data.filename}`, 'success');
        } catch (err) {
            badge.innerHTML = `
                <div class="uploaded-file-icon">❌</div>
                <div class="uploaded-file-info">
                    <div class="uploaded-file-name">${escapeHtml(file.name)}</div>
                    <div class="uploaded-file-meta" style="color: var(--danger);">${err.message}</div>
                </div>
                <button class="btn-remove" title="Remove">&times;</button>
            `;
            badge.querySelector('.btn-remove').addEventListener('click', () => badge.remove());
            showToast(`Failed to process ${file.name}: ${err.message}`, 'error');
        }
    }
});

// Upload PDF for explain section
setupDropZone('explainDropZone', 'explainFileInput', 'explainBrowseBtn', async (files) => {
    const file = files[0]; // Single file for explain
    const badge = document.getElementById('explainFileBadge');
    const textarea = document.getElementById('explainResumeInput');

    badge.style.display = 'inline-flex';
    badge.innerHTML = `<span class="upload-spinner"></span> Processing ${escapeHtml(file.name)}...`;

    try {
        const formData = new FormData();
        formData.append('file', file);

        const res = await fetch(`${API_BASE}/upload-pdf`, { method: 'POST', body: formData });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Upload failed');
        }

        const data = await res.json();
        textarea.value = data.text;
        badge.innerHTML = `✅ ${escapeHtml(data.filename)} — ${data.word_count} words`;
        showToast(`Loaded ${data.filename}`, 'success');
    } catch (err) {
        badge.innerHTML = `❌ Failed: ${err.message}`;
        badge.style.color = 'var(--danger)';
        showToast(`Failed: ${err.message}`, 'error');
    }
});

// ─── Resume Text Management ─────────────────────────────────────

let resumeCount = 1;

document.getElementById('addResumeBtn').addEventListener('click', () => {
    // Show text inputs if hidden
    const inputs = document.getElementById('resumeInputs');
    if (inputs.style.display === 'none') {
        inputs.style.display = 'flex';
        document.getElementById('toggleTextInput').textContent = 'Hide text input ▴';
    }

    resumeCount++;
    const entry = document.createElement('div');
    entry.className = 'resume-entry';
    entry.dataset.index = resumeCount;
    entry.innerHTML = `
        <div class="resume-entry-header">
            <input type="text" class="resume-name" placeholder="candidate_name" value="candidate_${resumeCount}">
            <button class="btn-remove" title="Remove">&times;</button>
        </div>
        <textarea class="text-input resume-text" placeholder="Paste resume text here..." rows="6"></textarea>
    `;

    entry.querySelector('.btn-remove').addEventListener('click', () => {
        entry.style.animation = 'fadeInUp 0.3s ease reverse';
        setTimeout(() => entry.remove(), 250);
    });

    inputs.appendChild(entry);
    inputs.scrollTop = inputs.scrollHeight;
});

// Remove button for first entry
document.querySelector('.resume-entry .btn-remove').addEventListener('click', function () {
    const entries = document.querySelectorAll('.resume-entry');
    if (entries.length > 1) {
        this.closest('.resume-entry').remove();
    }
});

// ─── Slider ─────────────────────────────────────────────────────

const slider = document.getElementById('topKSlider');
const sliderValue = document.getElementById('topKValue');

slider.addEventListener('input', () => {
    sliderValue.textContent = slider.value;
});

// ─── Rank Candidates ────────────────────────────────────────────

document.getElementById('rankBtn').addEventListener('click', async () => {
    const btn = document.getElementById('rankBtn');
    const jdText = document.getElementById('jdInput').value.trim();

    if (!jdText) {
        showToast('Please enter a job description.', 'error');
        return;
    }

    // Collect resumes from PDFs + text inputs
    const resumeTexts = { ...uploadedResumes };
    let hasResume = Object.keys(resumeTexts).length > 0;

    document.querySelectorAll('.resume-entry').forEach(entry => {
        const name = entry.querySelector('.resume-name').value.trim() || `candidate_${entry.dataset.index}`;
        const text = entry.querySelector('.resume-text').value.trim();
        if (text) {
            resumeTexts[name] = text;
            hasResume = true;
        }
    });

    if (!hasResume) {
        showToast('Please enter at least one resume.', 'error');
        return;
    }

    const topK = parseInt(slider.value);

    // Loading state
    btn.classList.add('loading');
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/rank`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                jd_text: jdText,
                resume_texts: resumeTexts,
                top_k: topK,
            }),
        });

        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        const data = await res.json();
        displayRankResults(data);
        showToast(`Ranked ${data.total_candidates} candidates successfully!`, 'success');
    } catch (err) {
        showToast(`Ranking failed: ${err.message}`, 'error');
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
});

function displayRankResults(data) {
    const container = document.getElementById('rankResults');
    const list = document.getElementById('resultsList');
    const count = document.getElementById('resultsCount');

    container.style.display = 'block';
    count.textContent = `${data.ranked_candidates.length} of ${data.total_candidates} shown`;
    list.innerHTML = '';

    const maxScore = data.ranked_candidates.length > 0
        ? Math.max(...data.ranked_candidates.map(r => Math.abs(r.score)))
        : 1;

    data.ranked_candidates.forEach((result, idx) => {
        const rankClass = idx === 0 ? 'gold' : idx === 1 ? 'silver' : idx === 2 ? 'bronze' : 'default';
        const medal = idx === 0 ? '🥇' : idx === 1 ? '🥈' : idx === 2 ? '🥉' : `#${result.rank}`;
        const barWidth = maxScore > 0 ? (Math.abs(result.score) / maxScore) * 100 : 0;

        const card = document.createElement('div');
        card.className = 'result-card';
        card.style.animationDelay = `${idx * 0.08}s`;

        card.innerHTML = `
            <div class="result-rank ${rankClass}">${medal}</div>
            <div class="result-info">
                <div class="result-name">${escapeHtml(result.filename)}</div>
                <div class="result-bar-bg">
                    <div class="result-bar-fill" style="width: 0%"></div>
                </div>
            </div>
            <div class="result-score">
                <div class="result-score-value">${(result.score * 100).toFixed(1)}</div>
                <div class="result-score-label">Match Score</div>
            </div>
        `;

        list.appendChild(card);

        // Animate bar
        requestAnimationFrame(() => {
            setTimeout(() => {
                card.querySelector('.result-bar-fill').style.width = `${barWidth}%`;
            }, 100 + idx * 80);
        });
    });

    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── Explain Match ──────────────────────────────────────────────

document.getElementById('explainBtn').addEventListener('click', async () => {
    const btn = document.getElementById('explainBtn');
    const jdText = document.getElementById('explainJdInput').value.trim();
    const resumeText = document.getElementById('explainResumeInput').value.trim();

    if (!jdText || !resumeText) {
        showToast('Please enter both a job description and a resume.', 'error');
        return;
    }

    btn.classList.add('loading');
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/explain`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                jd_text: jdText,
                resume_text: resumeText,
            }),
        });

        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        const data = await res.json();
        displayExplanation(data);
        showToast('Match explanation generated!', 'success');
    } catch (err) {
        showToast(`Explanation failed: ${err.message}`, 'error');
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
});

function displayExplanation(data) {
    const container = document.getElementById('explainResults');
    container.style.display = 'block';

    // Verdict
    const verdictEl = document.getElementById('verdictText');
    verdictEl.textContent = data.verdict || 'No verdict available';

    // Scores
    const scoresBars = document.getElementById('scoresBars');
    scoresBars.innerHTML = '';

    const scores = data.scores || {};
    for (const [name, value] of Object.entries(scores)) {
        const pct = Math.min(Math.abs(value) * 100, 100);
        const level = pct >= 60 ? 'high' : pct >= 35 ? 'medium' : 'low';
        const displayName = name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

        scoresBars.innerHTML += `
            <div class="metric-bar-item">
                <div class="metric-bar-label">
                    <span>${displayName}</span>
                    <span>${(value * 100).toFixed(1)}%</span>
                </div>
                <div class="metric-bar-track">
                    <div class="metric-bar-value ${level}" style="width: 0%"></div>
                </div>
            </div>
        `;

        requestAnimationFrame(() => {
            setTimeout(() => {
                const bars = scoresBars.querySelectorAll('.metric-bar-value');
                bars.forEach(bar => {
                    const label = bar.closest('.metric-bar-item').querySelector('.metric-bar-label span:last-child');
                    bar.style.width = `${parseFloat(label.textContent)}%`;
                });
            }, 200);
        });
    }

    // Skills
    const skillEl = document.getElementById('skillAnalysis');
    const skills = data.skill_analysis || {};
    skillEl.innerHTML = '';

    if (skills.matched_skills && skills.matched_skills.length > 0) {
        skillEl.innerHTML += `
            <div class="skill-group">
                <div class="skill-group-label">✅ Matched (${skills.matched_skills.length})</div>
                <div class="skill-tags">
                    ${skills.matched_skills.map((s, i) =>
            `<span class="skill-tag matched" style="animation-delay: ${i * 0.04}s">${escapeHtml(s)}</span>`
        ).join('')}
                </div>
            </div>
        `;
    }

    if (skills.missing_skills && skills.missing_skills.length > 0) {
        skillEl.innerHTML += `
            <div class="skill-group">
                <div class="skill-group-label">❌ Missing (${skills.missing_skills.length})</div>
                <div class="skill-tags">
                    ${skills.missing_skills.map((s, i) =>
            `<span class="skill-tag missing" style="animation-delay: ${i * 0.04}s">${escapeHtml(s)}</span>`
        ).join('')}
                </div>
            </div>
        `;
    }

    if (skills.extra_skills && skills.extra_skills.length > 0) {
        skillEl.innerHTML += `
            <div class="skill-group">
                <div class="skill-group-label">💡 Extra Skills (${skills.extra_skills.length})</div>
                <div class="skill-tags">
                    ${skills.extra_skills.slice(0, 12).map((s, i) =>
            `<span class="skill-tag extra" style="animation-delay: ${i * 0.04}s">${escapeHtml(s)}</span>`
        ).join('')}
                </div>
            </div>
        `;
    }

    if (skills.coverage !== undefined) {
        skillEl.innerHTML += `
            <div class="coverage-stat">
                <span>Skill Coverage:</span>
                <span class="coverage-value">${(skills.coverage * 100).toFixed(0)}%</span>
                <span>(${skills.match_ratio || '?'})</span>
            </div>
        `;
    }

    // Keywords
    const kwEl = document.getElementById('keywordAnalysis');
    const kw = data.keyword_overlap || {};
    kwEl.innerHTML = '';

    if (kw.shared_keywords && kw.shared_keywords.length > 0) {
        kwEl.innerHTML += `
            <div class="skill-group">
                <div class="skill-group-label">Shared Keywords</div>
                <div class="skill-tags">
                    ${kw.shared_keywords.slice(0, 15).map((s, i) =>
            `<span class="skill-tag keyword" style="animation-delay: ${i * 0.03}s">${escapeHtml(s)}</span>`
        ).join('')}
                </div>
            </div>
        `;
    }

    if (kw.overlap_ratio !== undefined) {
        kwEl.innerHTML += `
            <div class="coverage-stat">
                <span>Keyword Overlap:</span>
                <span class="coverage-value">${(kw.overlap_ratio * 100).toFixed(0)}%</span>
            </div>
        `;
    }

    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── Dashboard ──────────────────────────────────────────────────

async function loadDashboard() {
    try {
        // Fetch stats
        const [statsRes, cacheRes] = await Promise.all([
            fetch(`${API_BASE}/stats`).catch(() => null),
            fetch(`${API_BASE}/cache/stats`).catch(() => null),
        ]);

        if (statsRes && statsRes.ok) {
            const stats = await statsRes.json();

            animateNumber('statResumes', stats.resumes_processed || 0);
            animateNumber('statJds', stats.jds_balanced || 0);

            const totalPairs = Object.values(stats.pair_datasets || {}).reduce((a, b) => a + b, 0);
            animateNumber('statPairs', totalPairs);

            // Animate stat bars
            const maxVal = Math.max(stats.resumes_processed || 0, stats.jds_balanced || 0, totalPairs, 1);
            const statCards = document.querySelectorAll('.stat-card');
            const vals = [stats.resumes_processed || 0, stats.jds_balanced || 0, totalPairs, 0];
            statCards.forEach((card, i) => {
                setTimeout(() => {
                    const fill = card.querySelector('.stat-bar-fill');
                    if (fill) fill.style.width = `${(vals[i] / maxVal) * 100}%`;
                }, 300 + i * 150);
            });

            // Pair breakdown
            displayPairBreakdown(stats.pair_datasets || {});
        }

        if (cacheRes && cacheRes.ok) {
            const cache = await cacheRes.json();
            document.getElementById('statCache').textContent = `${cache.total_size_mb || 0} MB`;

            const statCards = document.querySelectorAll('.stat-card');
            setTimeout(() => {
                const fill = statCards[3]?.querySelector('.stat-bar-fill');
                if (fill) fill.style.width = `${Math.min((cache.total_size_mb / 100) * 100, 100)}%`;
            }, 750);
        }
    } catch (err) {
        showToast('Failed to load dashboard data', 'error');
    }
}

function animateNumber(elementId, target) {
    const el = document.getElementById(elementId);
    const duration = 1000;
    const start = performance.now();
    const startVal = 0;

    function update(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(startVal + (target - startVal) * eased);
        el.textContent = current.toLocaleString();
        if (progress < 1) requestAnimationFrame(update);
    }

    requestAnimationFrame(update);
}

function displayPairBreakdown(datasets) {
    const container = document.getElementById('pairBreakdown');
    container.innerHTML = '';

    const entries = Object.entries(datasets);
    if (entries.length === 0) {
        container.innerHTML = '<p style="color: var(--text-muted); font-size: 0.875rem;">No pair datasets found.</p>';
        return;
    }

    const maxCount = Math.max(...entries.map(([, v]) => v));

    entries.forEach(([name, count], idx) => {
        const displayName = name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        const barWidth = (count / maxCount) * 100;

        const item = document.createElement('div');
        item.className = 'pair-bar-item';
        item.innerHTML = `
            <div class="pair-bar-name">${displayName}</div>
            <div class="pair-bar-track">
                <div class="pair-bar-fill" style="width: 0%">
                    <span class="pair-bar-count">${count.toLocaleString()}</span>
                </div>
            </div>
        `;

        container.appendChild(item);

        setTimeout(() => {
            item.querySelector('.pair-bar-fill').style.width = `${barWidth}%`;
        }, 300 + idx * 200);
    });
}

// ─── Toast Notifications ────────────────────────────────────────

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const icons = { success: '✓', error: '✕', info: 'ℹ' };

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span style="font-weight: 700; font-size: 1.1rem;">${icons[type] || '•'}</span>
        <span>${escapeHtml(message)}</span>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('toast-exit');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ─── Utilities ──────────────────────────────────────────────────

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ─── Keyboard Shortcuts ─────────────────────────────────────────

document.addEventListener('keydown', (e) => {
    if (e.altKey) {
        if (e.key === '1') switchSection('rank');
        if (e.key === '2') switchSection('explain');
        if (e.key === '3') switchSection('dashboard');
    }
});
