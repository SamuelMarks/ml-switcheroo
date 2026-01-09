/**
 * trace_render.js
 *
 * Logic for rendering the "Git Graph" visualization of the transpilation process.
 * Consumes the JSON event stream from the ASTEngine.
 *
 * Update: Supports Time Travel debugging via TimeTravelController.
 */

class TraceGraph {
    /**
     * @param {string} containerId - The ID of the HTML element to render into.
     * @param {function} onHover - Callback (eventData, isHovering) => void
     */
    constructor(containerId, onHover = null) {
        this.container = document.getElementById(containerId);
        this.eventsMap = new Map();
        this.onHover = onHover;
    }

    /**
     * Main entry point.
     * @param {Array<Object>} events - List of TraceEvent objects.
     */
    render(events) {
        if (!this.container) return;
        this.container.innerHTML = "";
        this.container.style.display = "block";

        if (!events || events.length === 0) {
            this._renderPlaceholder("No trace events captured.");
            return;
        }

        // 1. Pre-process: Map IDs and Calculate Depth
        const hierarchy = this._buildHierarchy(events);

        // 2. Render Rows (Skipping noise like phase_end)
        hierarchy.forEach((item, index) => {
            if (item.type === 'phase_end') return;

            const isLast = (index === hierarchy.length - 1);
            const rowDOM = this._createRow(item, isLast);
            this.container.appendChild(rowDOM);
        });
    }

    _renderPlaceholder(msg) {
        this.container.innerHTML = `<div style="padding:20px; text-align:center; color:#888;">${msg}</div>`;
    }

    /**
     * Analyzes parent_id chains to assign depth levels.
     */
    _buildHierarchy(events) {
        // Index by ID
        events.forEach(e => this.eventsMap.set(e.id, e));

        return events.map(e => {
            let depth = 0;
            let parentId = e.parent_id;

            // Walk up the tree to count depth
            while (parentId && this.eventsMap.has(parentId)) {
                depth++;
                parentId = this.eventsMap.get(parentId).parent_id;
            }

            return {
                ...e,
                uiDepth: depth
            };
        });
    }

    _createRow(event, isLastEvent) {
        const row = document.createElement('div');
        row.className = `trace-row trace-depth-${event.uiDepth} type-${event.type}`;

        // --- Interaction Logic (Line Hover) ---
        if (this.onHover && event.lineno) {
            row.style.cursor = "crosshair";
            row.title = `Source Line: ${event.lineno}`; // Tooltip

            row.addEventListener('mouseenter', () => {
                this.onHover(event, true);
                row.style.backgroundColor = 'rgba(255, 249, 196, 0.3)'; // Subtle hover visual in list
            });

            row.addEventListener('mouseleave', () => {
                this.onHover(event, false);
                row.style.backgroundColor = 'transparent';
            });
        }
        
        // --- Timeline Seek Interaction ---
        if (event.type === 'ast_snapshot') {
            row.style.cursor = "pointer";
            row.title = "Click to jump to this state";
            row.addEventListener('click', () => {
                // Dispatch event for TimeTravelController
                const evt = new CustomEvent('tt-seek-request', { detail: { eventId: event.id } });
                document.dispatchEvent(evt);
            });
        }

        // --- 1. Timestamp Column ---
        const meta = document.createElement('div');
        meta.className = 'trace-meta';
        if (event.timestamp) {
            const date = new Date(event.timestamp * 1000);
            meta.textContent = date.toISOString().split("T")[1].substring(0, 8); // HH:MM:SS
        }

        // --- 2. Graph Column (Rails) ---
        const graphCol = document.createElement('div');
        graphCol.className = 'trace-graph-col';

        // The Rail (Vertical Line)
        const rail = document.createElement('div');
        rail.className = 'trace-rail';

        // The Node (Dot)
        const node = document.createElement('div');
        node.className = `trace-node`;

        graphCol.appendChild(rail);
        graphCol.appendChild(node);

        // --- 3. Content Column ---
        const content = document.createElement('div');
        content.className = 'trace-content';

        // Title Line
        const header = document.createElement('div');
        header.className = 'trace-header';

        // Optional Tag
        if (event.type === 'phase_start') {
            header.innerHTML += `<span class="trace-tag tag-phase">PHASE</span>`;
        } else if (event.type === 'match_semantics') {
            header.innerHTML += `<span class="trace-tag tag-match">MATCH</span>`;
        } else if (event.type === 'analysis_warning') {
            header.innerHTML += `<span class="trace-tag tag-warn">WARN</span>`;
        } else if (event.type === 'inspection') {
            header.innerHTML += `<span class="trace-tag tag-inspect">SCAN</span>`;
        } else if (event.type === 'ast_snapshot') {
            header.innerHTML += `<span class="trace-tag tag-snap">STATE</span>`;
        }

        const titleText = document.createElement('span');
        titleText.className = 'trace-title-text';
        titleText.textContent = event.description;
        header.appendChild(titleText);
        content.appendChild(header);

        // Metadata Rendering (Details)
        if (event.metadata && Object.keys(event.metadata).length > 0) {
            const details = document.createElement('div');
            details.className = 'trace-details';

            // Special Case: "detail" string (Phase description)
            if (event.metadata.detail) {
                const p = document.createElement('div');
                p.className = 'detail-text';
                p.textContent = event.metadata.detail;
                details.appendChild(p);
            }

            // Special Case: Match Logic (Source -> Target)
            if (event.metadata.source && event.metadata.target) {
                const mapInfo = document.createElement('div');
                mapInfo.className = 'detail-map';
                mapInfo.innerHTML = `
                    <span class="map-pill">${event.metadata.source}</span>
                    <span class="map-arrow">➔</span>
                    <span class="map-pill">${event.metadata.target}</span>
                `;
                details.appendChild(mapInfo);
            }

            // Special Case: Outcome (for Inspection)
            if (event.metadata.outcome) {
                const info = document.createElement('div');
                info.className = 'detail-text';
                info.innerHTML = `<strong>Outcome:</strong> ${event.metadata.outcome}`;
                if (event.metadata.detail) {
                    info.innerHTML += ` &mdash; ${event.metadata.detail}`;
                }
                details.appendChild(info);
            }

            // Special Case: Code Diffs
            if (event.metadata.before || event.metadata.after) {
                const diffBox = this._createDiffBox(event.metadata.before, event.metadata.after);
                details.appendChild(diffBox);
            }

            content.appendChild(details);
        }

        row.appendChild(meta);
        row.appendChild(graphCol);
        row.appendChild(content);

        return row;
    }

    _createDiffBox(codeBefore, codeAfter) {
        const box = document.createElement('div');
        box.className = 'trace-diff-box';

        // Comparison Grid
        const grid = document.createElement('div');
        grid.className = 'diff-grid';

        const left = document.createElement('div');
        left.className = 'diff-col diff-del';
        left.textContent = (codeBefore || "").trim();

        const right = document.createElement('div');
        right.className = 'diff-col diff-add';
        right.textContent = (codeAfter || "").trim();

        grid.appendChild(left);
        grid.appendChild(right);
        box.appendChild(grid);

        return box;
    }
}

/**
 * Controller for Time Travel Debugging.
 * Indexes AST_SNAPSHOT events and manages UI state (slider, buttons).
 */
class TimeTravelController {
    /**
     * @param {Object} callbacks - Functions to update the UI.
     * @param {function} callbacks.onUpdateCode - (codeStr) => void
     * @param {function} callbacks.onUpdateGraph - (mermaidStr) => void
     */
    constructor(callbacks) {
        this.snapshots = [];
        this.currentIndex = -1;
        this.callbacks = callbacks;

        // UI Elements
        this.slider = document.getElementById("tt-slider");
        this.btnPrev = document.getElementById("btn-tt-prev");
        this.btnNext = document.getElementById("btn-tt-next");
        this.label = document.getElementById("tt-phase-label");

        this._bindEvents();
        
        // Listen for timeline clicks
        document.addEventListener('tt-seek-request', (e) => {
            if (e.detail && e.detail.eventId) {
                this.seekToId(e.detail.eventId);
            }
        });
    }
    
    _bindEvents() {
        if (!this.slider) return;

        this.slider.addEventListener('input', (e) => {
            this.seek(parseInt(e.target.value));
        });

        if (this.btnPrev) {
            this.btnPrev.addEventListener('click', () => this.step(-1));
        }
        if (this.btnNext) {
            this.btnNext.addEventListener('click', () => this.step(1));
        }
    }

    /**
     * Loads a new trace event stream.
     * @param {Array} events - Full list of TraceEvents from engine.
     */
    loadEvents(events) {
        if (!events) return;
        
        // Filter for snapshots and sort by timestamp implicitly via list order
        this.snapshots = events.filter(e => e.type === "ast_snapshot");
        
        if (this.snapshots.length === 0) {
            this._resetUI(false);
            return;
        }

        // Configure Slider
        const maxIdx = this.snapshots.length - 1;
        if (this.slider) {
            this.slider.min = "0";
            this.slider.max = maxIdx.toString();
            this.slider.value = maxIdx.toString();
            this.slider.disabled = false;
        }

        this._resetUI(true);
        // Default to latest state
        this.seek(maxIdx);
    }
    
    _resetUI(enabled) {
        if (this.btnPrev) this.btnPrev.disabled = !enabled;
        if (this.btnNext) this.btnNext.disabled = !enabled;
        if (this.label) this.label.textContent = enabled ? "Ready" : "No Snapshots";
        if (!enabled && this.slider) this.slider.disabled = true;
    }

    /**
     * Steps forward or backward relative to current index.
     * @param {number} delta - +1 or -1.
     */
    step(delta) {
        this.seek(this.currentIndex + delta);
    }

    /**
     * Seeks to a specific index in the snapshot array.
     * @param {number} index - The target index.
     */
    seek(index) {
        if (index < 0 || index >= this.snapshots.length) return;
        
        this.currentIndex = index;
        const snap = this.snapshots[index];

        // Update UI
        if (this.slider) this.slider.value = index;
        if (this.label) this.label.textContent = `${index + 1}/${this.snapshots.length}: ${snap.description}`;
        
        if (this.btnPrev) this.btnPrev.disabled = (index === 0);
        if (this.btnNext) this.btnNext.disabled = (index === this.snapshots.length - 1);

        // Fire Callbacks
        if (snap.metadata) {
            if (this.callbacks.onUpdateCode && snap.metadata.code) {
                this.callbacks.onUpdateCode(snap.metadata.code);
            }
            if (this.callbacks.onUpdateGraph && snap.metadata.mermaid) {
                this.callbacks.onUpdateGraph(snap.metadata.mermaid);
            }
        }
    }
    
    /**
     * Seeks to the snapshot with the matching Event ID.
     * @param {string} id - The event uuid.
     */
    seekToId(id) {
        const idx = this.snapshots.findIndex(s => s.id === id);
        if (idx !== -1) {
            this.seek(idx);
        }
    }
}

// Export
window.TraceGraph = TraceGraph;
window.TimeTravelController = TimeTravelController;