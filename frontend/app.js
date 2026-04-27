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

let lastRankContext = { jdText: '', resumes: {}, ranked: [] };

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
        lastRankContext = {
            jdText: jdText,
            resumes: resumeTexts,
            ranked: data.ranked_candidates,
        };
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

        const wrapper = document.createElement('div');
        wrapper.className = 'result-item';

        wrapper.innerHTML = `
            <div class="result-card" style="animation-delay: ${idx * 0.08}s">
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
                <button class="btn-why-rank" data-idx="${idx}" type="button">
                    <span class="why-text">Why this rank?</span>
                    <span class="why-caret">▾</span>
                </button>
            </div>
            <div class="rank-explanation" id="rank-exp-${idx}"></div>
        `;

        list.appendChild(wrapper);

        // Animate bar
        requestAnimationFrame(() => {
            setTimeout(() => {
                wrapper.querySelector('.result-bar-fill').style.width = `${barWidth}%`;
            }, 100 + idx * 80);
        });

        // Why-this-rank toggle
        wrapper.querySelector('.btn-why-rank').addEventListener('click', (e) => {
            e.stopPropagation();
            toggleRankExplanation(idx);
        });
    });

    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function toggleRankExplanation(idx) {
    const expDiv = document.getElementById(`rank-exp-${idx}`);
    const btn = document.querySelector(`.btn-why-rank[data-idx="${idx}"]`);
    if (!expDiv || !btn) return;

    const textEl = btn.querySelector('.why-text');
    const caretEl = btn.querySelector('.why-caret');

    // Toggle visibility if already loaded
    if (expDiv.classList.contains('open')) {
        expDiv.classList.remove('open');
        textEl.textContent = 'Why this rank?';
        caretEl.textContent = '▾';
        return;
    }

    if (expDiv.dataset.loaded === 'true') {
        expDiv.classList.add('open');
        textEl.textContent = 'Hide explanation';
        caretEl.textContent = '▴';
        return;
    }

    // First load — fetch /explain
    const candidate = lastRankContext.ranked[idx];
    const resumeText = lastRankContext.resumes[candidate.filename];

    if (!resumeText) {
        showToast('Resume text not available for this candidate', 'error');
        return;
    }

    expDiv.classList.add('open');
    expDiv.innerHTML = `
        <div class="rank-exp-loading">
            <span class="upload-spinner"></span>
            Analyzing why this candidate ranked at #${candidate.rank}...
        </div>
    `;
    textEl.textContent = 'Hide explanation';
    caretEl.textContent = '▴';
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/explain`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                jd_text: lastRankContext.jdText,
                resume_text: resumeText,
            }),
        });
        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        const data = await res.json();

        renderRankExplanation(expDiv, idx, candidate, data);
        expDiv.dataset.loaded = 'true';
    } catch (err) {
        expDiv.innerHTML = `<div class="rank-exp-error">Failed to load explanation: ${escapeHtml(err.message)}</div>`;
    } finally {
        btn.disabled = false;
    }
}

function renderRankExplanation(container, idx, candidate, data) {
    const ranked = lastRankContext.ranked;
    const rank = candidate.rank;
    const score = candidate.score;
    const total = ranked.length;

    // ── Rank Context block ──
    let rankContextHtml;
    if (rank === 1) {
        const second = ranked[1];
        if (second) {
            const gap = (score - second.score) * 100;
            rankContextHtml = `
                <p class="reasoning-text">
                    🏆 <strong>Top-ranked of ${total} candidates.</strong> This resume scored
                    higher than every other submission. The closest competitor was
                    <strong>${escapeHtml(second.filename)}</strong> at #2, trailing by
                    <strong>${gap.toFixed(2)} points</strong>
                    (${(second.score * 100).toFixed(1)}%).
                </p>
            `;
        } else {
            rankContextHtml = `<p class="reasoning-text">🏆 <strong>Top-ranked</strong> — only one candidate was submitted.</p>`;
        }
    } else {
        const leader = ranked[0];
        const next = ranked[idx + 1];
        const gapToTop = (leader.score - score) * 100;
        const parts = [
            `Ranked <strong>#${rank} of ${total}</strong> with <strong>${(score * 100).toFixed(1)}%</strong>.`,
            `Behind <strong>#1 (${escapeHtml(leader.filename)})</strong> by <strong>${gapToTop.toFixed(2)} points</strong>.`,
        ];
        if (next) {
            const gapToNext = (score - next.score) * 100;
            parts.push(`Ahead of #${next.rank} (${escapeHtml(next.filename)}) by ${gapToNext.toFixed(2)} points.`);
        }
        rankContextHtml = `<p class="reasoning-text">${parts.join(' ')}</p>`;
    }

    // ── Reasoning data ──
    const sbert = (data.scores && data.scores.sbert_similarity) || 0;
    const pct = sbert * 100;
    const skillCov = (data.scores && data.scores.skill_coverage) || 0;
    const matched = (data.skill_analysis && data.skill_analysis.matched_skills) || [];
    const missing = (data.skill_analysis && data.skill_analysis.missing_skills) || [];
    const shared = (data.keyword_overlap && data.keyword_overlap.shared_keywords) || [];
    const jdOnly = (data.keyword_overlap && data.keyword_overlap.jd_only_keywords) || [];
    const overlapRatio = (data.keyword_overlap && data.keyword_overlap.overlap_ratio) || 0;

    const matchedSet = new Set(matched);
    const missingSet = new Set(missing);
    const boosters = [...matched, ...shared.filter(k => !matchedSet.has(k))].slice(0, 10);
    const drainers = [...missing, ...jdOnly.filter(k => !missingSet.has(k))].slice(0, 10);

    container.innerHTML = `
        <div class="reasoning-block">
            <div class="reasoning-label">📊 Rank Context</div>
            ${rankContextHtml}
        </div>

        <div class="reasoning-block">
            <div class="reasoning-label">🎯 Score Breakdown</div>
            <div class="rank-score-grid">
                <div class="rank-score-cell">
                    <div class="rank-score-cell-value">${pct.toFixed(1)}%</div>
                    <div class="rank-score-cell-label">Semantic Similarity</div>
                </div>
                <div class="rank-score-cell">
                    <div class="rank-score-cell-value">${(skillCov * 100).toFixed(0)}%</div>
                    <div class="rank-score-cell-label">Skill Coverage (${matched.length}/${matched.length + missing.length})</div>
                </div>
                <div class="rank-score-cell">
                    <div class="rank-score-cell-value">${(overlapRatio * 100).toFixed(0)}%</div>
                    <div class="rank-score-cell-label">Keyword Overlap</div>
                </div>
            </div>
        </div>

        ${boosters.length > 0 ? `
        <div class="reasoning-block">
            <div class="reasoning-label">⬆️ Why this candidate ranked here</div>
            <p class="reasoning-text">
                These concepts in the resume align with the JD and pulled the score up:
            </p>
            <div class="skill-tags">
                ${boosters.map(s => `<span class="skill-tag matched">${escapeHtml(s)}</span>`).join('')}
            </div>
        </div>` : ''}

        ${drainers.length > 0 ? `
        <div class="reasoning-block">
            <div class="reasoning-label">⬇️ What pulled the rank down</div>
            <p class="reasoning-text">
                These JD requirements are missing from the resume — each one widens
                the gap from the top:
            </p>
            <div class="skill-tags">
                ${drainers.map(s => `<span class="skill-tag missing">${escapeHtml(s)}</span>`).join('')}
            </div>
        </div>` : ''}

        <div class="reasoning-block reasoning-bottom-line">
            <div class="reasoning-label">💡 Bottom Line</div>
            <p class="reasoning-text">${generateRankBottomLine(rank, total, pct, matched.length, missing.length, ranked)}</p>
        </div>
    `;
}

function generateRankBottomLine(rank, total, pct, matchedCount, missingCount, ranked) {
    if (rank === 1) {
        const second = ranked[1];
        const gapStr = second ? ` and outscored the runner-up by ${((ranked[0].score - second.score) * 100).toFixed(1)} points` : '';
        return `This candidate placed first out of ${total} because the embedding model found the strongest semantic alignment between this resume and the JD (${pct.toFixed(1)}%), with ${matchedCount} matched skill${matchedCount !== 1 ? 's' : ''}${gapStr}. Other candidates either had fewer matched skills, more missing requirements, or used vocabulary that diverged from the JD's domain.`;
    } else if (rank === 2) {
        const leader = ranked[0];
        const gap = ((leader.score - ranked[rank - 1].score) * 100).toFixed(1);
        return `Strong runner-up at #${rank} of ${total}. Just <strong>${gap} points</strong> behind <strong>${escapeHtml(leader.filename)}</strong>. With ${matchedCount} matched skill${matchedCount !== 1 ? 's' : ''} this candidate is competitive — closing the gap would mean covering the ${missingCount} missing requirement${missingCount !== 1 ? 's' : ''} above.`;
    } else if (rank <= Math.ceil(total / 2)) {
        return `Ranked #${rank} of ${total} — mid-pack. The candidate covers part of the role with ${matchedCount} matched skill${matchedCount !== 1 ? 's' : ''}, but ${missingCount} JD requirement${missingCount !== 1 ? 's' : ''} not present in the resume kept this candidate from climbing higher.`;
    } else {
        return `Ranked #${rank} of ${total} — bottom half. With only ${pct.toFixed(1)}% semantic alignment and ${missingCount} missing skill${missingCount !== 1 ? 's' : ''}, this resume's professional vocabulary and domain sit further from the JD than the higher-ranked candidates.`;
    }
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

    // Semantic similarity reasoning
    displaySemanticReasoning(data);

    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displaySemanticReasoning(data) {
    const container = document.getElementById('semanticReasoning');
    if (!container) return;

    const sbert = (data.scores && data.scores.sbert_similarity) || 0;
    const pct = sbert * 100;

    let band, bandClass, bandText;
    if (pct >= 70) {
        band = 'Very Strong';
        bandClass = 'high';
        bandText = 'The resume is conceptually almost interchangeable with the job description. Both texts describe the same kind of role, domain, and seniority — the embeddings land in nearly the same region of vector space.';
    } else if (pct >= 55) {
        band = 'Strong';
        bandClass = 'high';
        bandText = 'The resume and JD share substantial conceptual ground. The model recognizes that the candidate works in the same domain and uses similar professional vocabulary, even if exact phrasing differs.';
    } else if (pct >= 40) {
        band = 'Moderate';
        bandClass = 'medium';
        bandText = 'There is partial conceptual overlap. The candidate likely covers some of the role\'s subject matter but operates in an adjacent specialty or at a different level — embeddings are close but not aligned.';
    } else if (pct >= 25) {
        band = 'Weak';
        bandClass = 'low';
        bandText = 'The texts share only loose thematic overlap. The candidate\'s background touches the JD\'s domain peripherally — common professional language exists, but core requirements are not represented.';
    } else {
        band = 'Very Weak';
        bandClass = 'low';
        bandText = 'The resume and JD describe essentially different things. The embeddings sit in distant regions of vector space — different domains, different responsibilities, or different seniority altogether.';
    }

    const matched = (data.skill_analysis && data.skill_analysis.matched_skills) || [];
    const missing = (data.skill_analysis && data.skill_analysis.missing_skills) || [];
    const shared = (data.keyword_overlap && data.keyword_overlap.shared_keywords) || [];
    const jdOnly = (data.keyword_overlap && data.keyword_overlap.jd_only_keywords) || [];

    const matchedSet = new Set(matched);
    const missingSet = new Set(missing);
    const boosters = [...matched, ...shared.filter(k => !matchedSet.has(k))].slice(0, 10);
    const drainers = [...missing, ...jdOnly.filter(k => !missingSet.has(k))].slice(0, 10);

    container.innerHTML = `
        <div class="reasoning-block">
            <div class="reasoning-label">📚 What is Semantic Similarity?</div>
            <p class="reasoning-text">
                Semantic similarity measures how closely the job description and resume align in
                <strong>meaning</strong> — not just shared words. Both texts are encoded by an SBERT
                transformer (<code>all-MiniLM-L6-v2</code>) into 384-dimensional vectors that capture
                conceptual content, then compared using <strong>cosine similarity</strong>.
                A score of 100% means identical meaning; 0% means unrelated.
            </p>
        </div>

        <div class="reasoning-block">
            <div class="reasoning-label">🎯 Score Interpretation</div>
            <div class="reasoning-score-row">
                <div class="reasoning-score-value">${pct.toFixed(1)}%</div>
                <div class="reasoning-score-band ${bandClass}">${band}</div>
            </div>
            <p class="reasoning-text">${bandText}</p>
        </div>

        ${boosters.length > 0 ? `
        <div class="reasoning-block">
            <div class="reasoning-label">⬆️ What's Boosting the Score</div>
            <p class="reasoning-text">
                These shared concepts appear in <em>both</em> the JD and the resume. They pull the
                two embeddings closer together in vector space:
            </p>
            <div class="skill-tags">
                ${boosters.map(s => `<span class="skill-tag matched">${escapeHtml(s)}</span>`).join('')}
            </div>
        </div>` : ''}

        ${drainers.length > 0 ? `
        <div class="reasoning-block">
            <div class="reasoning-label">⬇️ What's Holding It Back</div>
            <p class="reasoning-text">
                These terms appear in the JD but are <em>missing</em> from the resume. Each absent
                concept widens the gap between the two embeddings:
            </p>
            <div class="skill-tags">
                ${drainers.map(s => `<span class="skill-tag missing">${escapeHtml(s)}</span>`).join('')}
            </div>
        </div>` : ''}

        <div class="reasoning-block reasoning-bottom-line">
            <div class="reasoning-label">💡 Bottom Line</div>
            <p class="reasoning-text">${generateBottomLine(pct, matched.length, missing.length, shared.length)}</p>
        </div>
    `;
}

function generateBottomLine(pct, matchedCount, missingCount, sharedCount) {
    const score = pct.toFixed(1);
    if (pct >= 55) {
        return `The <strong>${score}%</strong> similarity reflects strong alignment between the resume and the role. The model recognizes ${matchedCount} matched skill${matchedCount !== 1 ? 's' : ''} and ${sharedCount} overlapping keyword${sharedCount !== 1 ? 's' : ''}, suggesting this candidate is a viable semantic fit. Lower-ranked aspects (missing skills, phrasing differences) prevent it from being higher.`;
    } else if (pct >= 40) {
        return `The <strong>${score}%</strong> similarity indicates partial alignment. While ${matchedCount} skill${matchedCount !== 1 ? 's are' : ' is'} matched, ${missingCount} key requirement${missingCount !== 1 ? 's are' : ' is'} still missing. The candidate covers part of the role but has notable gaps that the embedding model picks up on.`;
    } else if (pct >= 25) {
        return `The <strong>${score}%</strong> similarity reflects limited alignment. Only ${matchedCount} of the JD's required skill${matchedCount !== 1 ? 's' : ''} appear in the resume, and ${missingCount} are missing. The candidate's professional vocabulary and domain only loosely overlap with what the JD describes.`;
    } else {
        return `The <strong>${score}%</strong> similarity reflects very weak alignment. With ${matchedCount} matched skill${matchedCount !== 1 ? 's' : ''} and ${missingCount} missing, the resume's content sits in a different conceptual region from the JD. This candidate likely targets a different role, domain, or seniority level.`;
    }
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
