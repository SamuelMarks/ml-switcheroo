/**
 * @fileoverview Git-graph style visualization of the ASTEngine transpilation
 * event stream and interactive Time Travel controller.
 */

/**
 * Renders the hierarchical transpilation trace event graph.
 */
class TraceGraph {
    /**
     * Initializes the `TraceGraph` renderer.
     *
     * @param {string} containerId - DOM element ID to render the trace list into.
     * @param {function(Object, boolean): void|null} onHover - Line highlight callback `(eventData, isHovering) => void`.
     */
    constructor(containerId, onHover = null) {
        this.container = document.getElementById(containerId);
        this.eventsMap = new Map();
        this.onHover = onHover;
    }

    /**
     * Renders a list of `TraceEvent` objects into the container.
     *
     * @param {Array<Object>} events - Array of trace events from `ASTEngine`.
     * @returns {void}
     */
    render(events) {
        if (!this.container) return;
        this.container.innerHTML = "";
        this.container.style.display = "block";

        if (!events || events.length === 0) {
            this._renderPlaceholder("No trace events captured.");
            return;
        }

        const hierarchy = this._buildHierarchy(events);

        hierarchy.forEach((item, index) => {
            if (item.type === 'phase_end') return;

            const isLast = (index === hierarchy.length - 1);
            const rowDOM = this._createRow(item, isLast);
            this.container.appendChild(rowDOM);
        });
    }

    /**
     * Renders an empty-state placeholder message inside the trace container.
     *
     * @param {string} msg - Placeholder message text.
     * @returns {void}
     */
    _renderPlaceholder(msg) {
        this.container.innerHTML = `<div style="padding:20px; text-align:center; color:#888;">${msg}</div>`;
    }

    /**
     * Computes `uiDepth` for each trace event by walking `parent_id` chains.
     *
     * @param {Array<Object>} events - Raw trace events.
     * @returns {Array<Object>} Trace events enriched with `uiDepth`.
     */
    _buildHierarchy(events) {
        events.forEach(e => this.eventsMap.set(e.id, e));

        return events.map(e => {
            let depth = 0;
            let parentId = e.parent_id;

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

    /**
     * Constructs a DOM row element for a single trace event.
     *
     * @param {Object} event - Enriched trace event object.
     * @param {boolean} _isLastEvent - Whether this is the final event in the list.
     * @returns {HTMLDivElement} Constructed trace row element.
     */
    _createRow(event, _isLastEvent) {
        const row = document.createElement('div');
        row.className = `trace-row trace-depth-${event.uiDepth} type-${event.type}`;

        if (this.onHover && event.lineno) {
            row.style.cursor = "crosshair";
            row.title = `Source Line: ${event.lineno}`;

            row.addEventListener('mouseenter', () => {
                this.onHover(event, true);
                row.style.backgroundColor = 'rgba(255, 249, 196, 0.3)';
            });

            row.addEventListener('mouseleave', () => {
                this.onHover(event, false);
                row.style.backgroundColor = 'transparent';
            });
        }

        if (event.type === 'ast_snapshot') {
            row.style.cursor = "pointer";
            row.title = "Click to jump to this state";
            row.addEventListener('click', () => {
                const evt = new CustomEvent('tt-seek-request', { detail: { eventId: event.id } });
                document.dispatchEvent(evt);
            });
        }

        const meta = document.createElement('div');
        meta.className = 'trace-meta';
        if (event.timestamp) {
            const date = new Date(event.timestamp * 1000);
            meta.textContent = date.toISOString().split("T")[1].substring(0, 8);
        }

        const graphCol = document.createElement('div');
        graphCol.className = 'trace-graph-col';

        const rail = document.createElement('div');
        rail.className = 'trace-rail';

        const node = document.createElement('div');
        node.className = `trace-node`;

        graphCol.appendChild(rail);
        graphCol.appendChild(node);

        const content = document.createElement('div');
        content.className = 'trace-content';

        const header = document.createElement('div');
        header.className = 'trace-header';

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

        if (event.metadata && Object.keys(event.metadata).length > 0) {
            const details = document.createElement('div');
            details.className = 'trace-details';

            if (event.metadata.detail) {
                const p = document.createElement('div');
                p.className = 'detail-text';
                p.textContent = event.metadata.detail;
                details.appendChild(p);
            }

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

            if (event.metadata.outcome) {
                const info = document.createElement('div');
                info.className = 'detail-text';
                info.innerHTML = `<strong>Outcome:</strong> ${event.metadata.outcome}`;
                if (event.metadata.detail) {
                    info.innerHTML += ` &mdash; ${event.metadata.detail}`;
                }
                details.appendChild(info);
            }

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

    /**
     * Creates a side-by-side before/after code diff element.
     *
     * @param {string} codeBefore - Source snippet before mutation.
     * @param {string} codeAfter - Source snippet after mutation.
     * @returns {HTMLDivElement} Diff container element.
     */
    _createDiffBox(codeBefore, codeAfter) {
        const box = document.createElement('div');
        box.className = 'trace-diff-box';

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
 * Controller for Time Travel Debugging across AST snapshots.
 */
class TimeTravelController {
    /**
     * Initializes the `TimeTravelController`.
     *
     * @param {{onUpdateCode?: function(string): void, onUpdateGraph?: function(string): void}} callbacks - UI update callbacks.
     */
    constructor(callbacks) {
        this.snapshots = [];
        this.currentIndex = -1;
        this.callbacks = callbacks;

        this.slider = document.getElementById("tt-slider");
        this.btnPrev = document.getElementById("btn-tt-prev");
        this.btnNext = document.getElementById("btn-tt-next");
        this.label = document.getElementById("tt-phase-label");

        this._bindEvents();

        document.addEventListener('tt-seek-request', (e) => {
            if (e.detail && e.detail.eventId) {
                this.seekToId(e.detail.eventId);
            }
        });
    }

    /**
     * Binds slider and step button DOM listeners.
     *
     * @returns {void}
     */
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
     * Loads a new trace event stream and seeks to the final snapshot.
     *
     * @param {Array<Object>} events - Full list of `TraceEvent` objects.
     * @returns {void}
     */
    loadEvents(events) {
        if (!events) return;

        this.snapshots = events.filter(e => e.type === "ast_snapshot");

        if (this.snapshots.length === 0) {
            this._resetUI(false);
            return;
        }

        const maxIdx = this.snapshots.length - 1;
        if (this.slider) {
            this.slider.min = "0";
            this.slider.max = maxIdx.toString();
            this.slider.value = maxIdx.toString();
            this.slider.disabled = false;
        }

        this._resetUI(true);
        this.seek(maxIdx);
    }

    /**
     * Enables or disables the time-travel controls.
     *
     * @param {boolean} enabled - Whether snapshots are available.
     * @returns {void}
     */
    _resetUI(enabled) {
        if (this.btnPrev) this.btnPrev.disabled = !enabled;
        if (this.btnNext) this.btnNext.disabled = !enabled;
        if (this.label) this.label.textContent = enabled ? "Ready" : "No Snapshots";
        if (!enabled && this.slider) this.slider.disabled = true;
    }

    /**
     * Steps forward or backward relative to `currentIndex`.
     *
     * @param {number} delta - Step offset (+1 or -1).
     * @returns {void}
     */
    step(delta) {
        this.seek(this.currentIndex + delta);
    }

    /**
     * Seeks to a specific index in `snapshots`.
     *
     * @param {number} index - Target snapshot index.
     * @returns {void}
     */
    seek(index) {
        if (index < 0 || index >= this.snapshots.length) return;

        this.currentIndex = index;
        const snap = this.snapshots[index];

        if (this.slider) this.slider.value = index;
        if (this.label) this.label.textContent = `${index + 1}/${this.snapshots.length}: ${snap.description}`;

        if (this.btnPrev) this.btnPrev.disabled = (index === 0);
        if (this.btnNext) this.btnNext.disabled = (index === this.snapshots.length - 1);

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
     * Seeks to the snapshot matching a specific event ID.
     *
     * @param {string} id - Event UUID string.
     * @returns {void}
     */
    seekToId(id) {
        const idx = this.snapshots.findIndex(s => s.id === id);
        if (idx !== -1) {
            this.seek(idx);
        }
    }
}

window.TraceGraph = TraceGraph;
window.TimeTravelController = TimeTravelController;
