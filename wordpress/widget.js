/*!
 * Ask Alcor - embeddable chat widget
 *
 * Usage (before </body>):
 *   <script src="widget.js" data-api-url="https://your-backend/api/chat"></script>
 *
 * Backend contract:
 *   POST { message, history: [{ role, content }] }
 *   ->   { answer, sources?: [{ title, url }] }
 *   GET  /health  (optional, used to wake a sleeping backend)
 */
(function () {
    "use strict";

    if (window.__askAlcorLoaded) return;
    window.__askAlcorLoaded = true;

    /* ==================================================================
     * 1. CONFIGURATION
     * ================================================================== */

    const currentScript = document.currentScript;

    const API_URL =
        (currentScript && currentScript.getAttribute("data-api-url")) ||
        window.ALCOR_API_URL ||
        "http://127.0.0.1:8000/api/chat";

    const HEALTH_URL = new URL("/health", new URL(API_URL, window.location.href)).href;

    const CONFIG = {
        // Example questions shown inside the empty input box
        hints: [
            "What does Alcor do?",
            "Who is the CEO of Alcor?",
            "Tell me about Alcor's leadership",
            "What services does Alcor offer?"
        ],
        hintIntervalMs: 2500,

        // Welcome message
        welcomeTitle: "Welcome to Ask Alcor",
        welcomeLead: "I'm the Alcor Solutions assistant. I can help you with:",
        welcomePoints: [
            "What Alcor does and the services it offers",
            "Alcor's CEO and leadership team",
            "Other company information in the Alcor knowledge base"
        ],

        // Memory
        historyStored: 20,
        historySent: 10,

        // Cold-start handling
        maxAttempts: 12,
        attemptTimeoutMs: 90000,
        totalDeadlineMs: 180000,
        slowNoticeAfterMs: 4000
    };

    /* ==================================================================
     * 2. STYLES
     * ================================================================== */

    const STYLES = `
        #ask-alcor-widget,
        #ask-alcor-widget * { box-sizing: border-box; }

        #ask-alcor-widget {
            --ac-ink: #0f1b2d;
            --ac-ink-2: #1d2b42;
            --ac-accent: #2f5fd0;
            --ac-accent-dark: #244db0;
            --ac-bg: #f5f7fa;
            --ac-line: #e2e7ee;
            --ac-text: #1f2937;
            --ac-muted: #6b7686;

            font-family: Inter, -apple-system, BlinkMacSystemFont,
                "Segoe UI", Roboto, Arial, sans-serif;
            font-size: 14px;
            line-height: 1.5;
            color: var(--ac-text);
            text-align: left;
            letter-spacing: normal;
            -webkit-font-smoothing: antialiased;
        }

        /* Reset: keep the host page's own styles from leaking in */
        #ask-alcor-widget p,
        #ask-alcor-widget ul,
        #ask-alcor-widget li {
            margin: 0;
            padding: 0;
            font-family: inherit;
            font-size: inherit;
            font-weight: inherit;
            line-height: inherit;
            color: inherit;
            letter-spacing: normal;
            text-transform: none;
        }

        #ask-alcor-widget button,
        #ask-alcor-widget input {
            margin: 0;
            font-family: inherit;
            letter-spacing: normal;
            text-transform: none;
        }

        #ask-alcor-widget button:focus-visible {
            outline: 3px solid rgba(47, 95, 208, 0.55);
            outline-offset: 2px;
        }

        /* ---------- Launcher ---------- */

        #ask-alcor-widget .alcor-launcher {
            position: fixed;
            right: 24px;
            bottom: 24px;
            z-index: 2147483000;

            display: flex;
            align-items: center;
            gap: 10px;

            height: 56px;
            padding: 0 20px 0 16px;

            border: none;
            border-radius: 28px;
            background: var(--ac-ink);
            color: #ffffff;
            cursor: pointer;

            font-size: 15px;
            font-weight: 600;

            box-shadow: 0 10px 30px rgba(15, 27, 45, 0.28);
            transition: background 0.2s, transform 0.2s, box-shadow 0.2s;
        }

        #ask-alcor-widget .alcor-launcher:hover {
            background: var(--ac-ink-2);
            transform: translateY(-1px);
            box-shadow: 0 14px 34px rgba(15, 27, 45, 0.32);
        }

        #ask-alcor-widget .alcor-launcher svg { width: 24px; height: 24px; flex: none; }
        #ask-alcor-widget .alcor-launcher .ico-close { display: none; }

        #ask-alcor-widget.is-open .alcor-launcher .ico-chat { display: none; }
        #ask-alcor-widget.is-open .alcor-launcher .ico-close { display: block; }
        #ask-alcor-widget.is-open .alcor-launcher-label { display: none; }

        #ask-alcor-widget.is-open .alcor-launcher {
            width: 56px;
            padding: 0;
            justify-content: center;
        }

        /* ---------- Chat window ---------- */

        #ask-alcor-widget .alcor-chat {
            position: fixed;
            right: 24px;
            bottom: 92px;
            z-index: 2147483000;

            width: 392px;
            max-width: calc(100vw - 32px);
            height: 610px;
            max-height: calc(100vh - 116px);

            display: flex;
            flex-direction: column;

            background: #ffffff;
            border: 1px solid var(--ac-line);
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 24px 64px rgba(15, 27, 45, 0.22);

            opacity: 0;
            visibility: hidden;
            transform: translateY(14px) scale(0.98);
            transform-origin: bottom right;
            transition: opacity 0.22s ease, transform 0.22s ease,
                visibility 0s linear 0.22s;
        }

        #ask-alcor-widget.is-open .alcor-chat {
            opacity: 1;
            visibility: visible;
            transform: none;
            transition: opacity 0.22s ease, transform 0.22s ease;
        }

        /* ---------- Header ---------- */

        #ask-alcor-widget .alcor-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-shrink: 0;
            padding: 16px 18px;
            background: var(--ac-ink);
            color: #ffffff;
        }

        #ask-alcor-widget .alcor-brand { display: flex; align-items: center; gap: 12px; }

        #ask-alcor-widget .alcor-logo {
            width: 38px;
            height: 38px;
            border-radius: 10px;
            background: #ffffff;
            color: var(--ac-ink);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 17px;
            font-weight: 800;
        }

        #ask-alcor-widget .alcor-title { font-size: 16px; font-weight: 700; line-height: 1.2; }

        #ask-alcor-widget .alcor-status {
            display: flex;
            align-items: center;
            gap: 6px;
            margin-top: 3px;
            font-size: 12px;
            color: #b8c2d2;
        }

        #ask-alcor-widget .alcor-status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #34d399;
        }

        #ask-alcor-widget .alcor-header-actions { display: flex; align-items: center; gap: 4px; }

        #ask-alcor-widget .alcor-new-chat,
        #ask-alcor-widget .alcor-close {
            border: none;
            background: transparent;
            color: #cbd5e1;
            cursor: pointer;
            border-radius: 8px;
            transition: background 0.15s, color 0.15s;
        }

        #ask-alcor-widget .alcor-new-chat { padding: 8px 10px; font-size: 12.5px; font-weight: 600; }

        #ask-alcor-widget .alcor-close {
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        #ask-alcor-widget .alcor-close svg { width: 18px; height: 18px; }

        #ask-alcor-widget .alcor-new-chat:hover,
        #ask-alcor-widget .alcor-close:hover {
            background: rgba(255, 255, 255, 0.12);
            color: #ffffff;
        }

        /* ---------- Messages ---------- */

        #ask-alcor-widget .alcor-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px 18px;
            background: var(--ac-bg);
            scroll-behavior: smooth;
        }

        #ask-alcor-widget .alcor-message {
            display: flex;
            margin-bottom: 14px;
            animation: alcorPop 0.25s ease both;
        }

        #ask-alcor-widget .alcor-message.bot  { justify-content: flex-start; }
        #ask-alcor-widget .alcor-message.user { justify-content: flex-end; }

        @keyframes alcorPop {
            from { opacity: 0; transform: translateY(8px); }
            to   { opacity: 1; transform: none; }
        }

        #ask-alcor-widget .alcor-bubble {
            max-width: 88%;
            padding: 12px 15px;
            border-radius: 14px;
            font-size: 14px;
            line-height: 1.6;
            word-wrap: break-word;
            overflow-wrap: anywhere;
        }

        #ask-alcor-widget .bot .alcor-bubble {
            background: #ffffff;
            color: var(--ac-text);
            border: 1px solid var(--ac-line);
            border-bottom-left-radius: 4px;
        }

        #ask-alcor-widget .user .alcor-bubble {
            background: var(--ac-accent);
            color: #ffffff;
            border-bottom-right-radius: 4px;
        }

        /* ---------- Answer formatting ---------- */

        #ask-alcor-widget .alcor-bubble p { margin: 0 0 8px; }
        #ask-alcor-widget .alcor-bubble p:last-child { margin-bottom: 0; }
        #ask-alcor-widget .alcor-bubble strong { font-weight: 700; }

        #ask-alcor-widget .alcor-bubble p.alcor-welcome-title {
            margin-bottom: 4px;
            font-weight: 700;
            color: var(--ac-ink);
        }

        #ask-alcor-widget .alcor-bubble p.alcor-welcome-lead { margin-bottom: 8px; }

        #ask-alcor-widget .alcor-bubble ul {
            margin: 6px 0 4px;
            padding: 0;
            list-style: none;
        }

        #ask-alcor-widget .alcor-bubble li {
            position: relative;
            margin-bottom: 4px;
            padding-left: 18px;
        }

        #ask-alcor-widget .alcor-bubble li::before {
            content: "";
            position: absolute;
            left: 2px;
            top: 0.62em;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--ac-accent);
        }

        /* ---------- Sources ---------- */

        #ask-alcor-widget .alcor-sources {
            margin-top: 12px;
            padding-top: 9px;
            border-top: 1px solid var(--ac-line);
        }

        #ask-alcor-widget .alcor-sources-title {
            margin-bottom: 5px;
            font-size: 11.5px;
            font-weight: 700;
            color: var(--ac-muted);
        }

        #ask-alcor-widget .alcor-source {
            display: block;
            margin-top: 4px;
            font-size: 12px;
            color: var(--ac-accent);
            text-decoration: none;
        }

        #ask-alcor-widget .alcor-source:hover { text-decoration: underline; }

        /* ---------- Typing + wait note ---------- */

        #ask-alcor-widget .alcor-typing { display: flex; align-items: center; gap: 5px; height: 20px; }

        #ask-alcor-widget .alcor-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #9aa5b5;
            animation: alcorTyping 1.2s infinite;
        }

        #ask-alcor-widget .alcor-dot:nth-child(2) { animation-delay: 0.15s; }
        #ask-alcor-widget .alcor-dot:nth-child(3) { animation-delay: 0.30s; }

        @keyframes alcorTyping {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.35; }
            30%           { transform: translateY(-3px); opacity: 1; }
        }

        #ask-alcor-widget .alcor-wait-note {
            margin-top: 6px;
            font-size: 12.5px;
            color: var(--ac-muted);
        }

        /* ---------- Error + retry ---------- */

        #ask-alcor-widget .alcor-retry {
            margin-top: 10px;
            padding: 8px 14px;
            border: 1px solid var(--ac-accent);
            border-radius: 8px;
            background: #ffffff;
            color: var(--ac-accent);
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.15s, color 0.15s;
        }

        #ask-alcor-widget .alcor-retry:hover { background: var(--ac-accent); color: #ffffff; }

        /* ---------- Input ---------- */

        #ask-alcor-widget .alcor-input-area {
            display: flex;
            gap: 10px;
            flex-shrink: 0;
            padding: 14px 16px;
            background: #ffffff;
            border-top: 1px solid var(--ac-line);
        }

        #ask-alcor-widget .alcor-field { position: relative; flex: 1; min-width: 0; }

        #ask-alcor-widget .alcor-input {
            width: 100%;
            height: 46px;
            padding: 0 14px;
            border: 1px solid #d3dae4;
            border-radius: 12px;
            outline: none;
            background: #ffffff;
            color: var(--ac-text);
            font-size: 14px;
            transition: border-color 0.15s, box-shadow 0.15s;
        }

        #ask-alcor-widget .alcor-input:focus {
            border-color: var(--ac-accent);
            box-shadow: 0 0 0 3px rgba(47, 95, 208, 0.14);
        }

        #ask-alcor-widget .alcor-input:disabled { background: #f3f5f8; }

        /* Rotating example questions */
        #ask-alcor-widget .alcor-hint {
            position: absolute;
            left: 15px;
            right: 12px;
            top: 50%;
            height: 22px;
            margin-top: -11px;

            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
            pointer-events: none;

            font-size: 14px;
            line-height: 22px;
            color: #8a94a4;

            opacity: 1;
            transform: translateY(0);
            transition: opacity 0.25s ease, transform 0.25s ease;
        }

        #ask-alcor-widget .alcor-hint b { font-weight: 600; color: #3b4a61; }
        #ask-alcor-widget .alcor-hint.is-swapping { opacity: 0; transform: translateY(6px); }
        #ask-alcor-widget .alcor-hint.is-hidden { opacity: 0; visibility: hidden; }

        #ask-alcor-widget .alcor-send {
            height: 46px;
            padding: 0 18px;
            border: none;
            border-radius: 12px;
            background: var(--ac-accent);
            color: #ffffff;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: background 0.15s;
        }

        #ask-alcor-widget .alcor-send:hover:not(:disabled) { background: var(--ac-accent-dark); }
        #ask-alcor-widget .alcor-send:disabled { opacity: 0.5; cursor: not-allowed; }

        #ask-alcor-widget .alcor-footer {
            padding: 0 16px 12px;
            background: #ffffff;
            text-align: center;
            font-size: 11.5px;
            color: #98a2b3;
        }

        /* ---------- Mobile ---------- */

        @media (max-width: 520px) {
            #ask-alcor-widget .alcor-launcher { right: 16px; bottom: 16px; }

            #ask-alcor-widget .alcor-chat {
                right: 0;
                bottom: 0;
                width: 100vw;
                max-width: 100vw;
                height: 100%;
                max-height: 100%;
                border: none;
                border-radius: 0;
            }

            #ask-alcor-widget.is-open .alcor-launcher { display: none; }
            #ask-alcor-widget .alcor-bubble { max-width: 92%; }
        }

        @media (prefers-reduced-motion: reduce) {
            #ask-alcor-widget * {
                animation: none !important;
                transition: none !important;
                scroll-behavior: auto !important;
            }
        }
    `;

    /* ==================================================================
     * 3. TEMPLATE
     * ================================================================== */

    const TEMPLATE = `
        <button class="alcor-launcher" type="button"
                aria-label="Open Ask Alcor chat" aria-expanded="false">
            <svg class="ico-chat" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="1.9"
                 stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M21 12a8 8 0 0 1-11.6 7.1L4 20.5l1.4-4.6A8 8 0 1 1 21 12z"/>
            </svg>
            <svg class="ico-close" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2"
                 stroke-linecap="round" aria-hidden="true">
                <path d="M6 6l12 12M18 6L6 18"/>
            </svg>
            <span class="alcor-launcher-label">Ask Alcor</span>
        </button>

        <section class="alcor-chat" role="dialog" aria-label="Ask Alcor chatbot">

            <header class="alcor-header">
                <div class="alcor-brand">
                    <div class="alcor-logo">A</div>
                    <div>
                        <div class="alcor-title">Ask Alcor</div>
                        <div class="alcor-status">
                            <span class="alcor-status-dot"></span>
                            Alcor Solutions AI Assistant
                        </div>
                    </div>
                </div>

                <div class="alcor-header-actions">
                    <button class="alcor-new-chat" type="button"
                            title="Start a new conversation">New chat</button>
                    <button class="alcor-close" type="button" aria-label="Close chat">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                             stroke-width="2" stroke-linecap="round" aria-hidden="true">
                            <path d="M6 6l12 12M18 6L6 18"/>
                        </svg>
                    </button>
                </div>
            </header>

            <div class="alcor-messages" aria-live="polite"></div>

            <form class="alcor-input-area" autocomplete="off">
                <div class="alcor-field">
                    <input class="alcor-input" type="text" maxlength="1000"
                           autocomplete="off" aria-label="Type your question" />
                    <div class="alcor-hint" aria-hidden="true"></div>
                </div>
                <button class="alcor-send" type="submit">Send</button>
            </form>

            <div class="alcor-footer">
                Answers are based on the Alcor knowledge base.
            </div>

        </section>
    `;

    /* ==================================================================
     * 4. MOUNT
     * ================================================================== */

    const styleEl = document.createElement("style");
    styleEl.setAttribute("data-ask-alcor", "");
    styleEl.textContent = STYLES;
    document.head.appendChild(styleEl);

    const container = document.createElement("div");
    container.id = "ask-alcor-widget";
    container.innerHTML = TEMPLATE;
    document.body.appendChild(container);

    const $ = (selector) => container.querySelector(selector);

    const launcher = $(".alcor-launcher");
    const closeButton = $(".alcor-close");
    const newChatButton = $(".alcor-new-chat");
    const messages = $(".alcor-messages");
    const form = $(".alcor-input-area");
    const input = $(".alcor-input");
    const hintEl = $(".alcor-hint");
    const sendButton = $(".alcor-send");

    /* ==================================================================
     * 5. STATE
     * ================================================================== */

    let conversationHistory = [];
    let requestInProgress = false;
    let isOpen = false;
    let chatId = 0;            // bumps on "New chat" so stale replies are dropped
    let hintTimer = null;
    let hintIndex = 0;
    let warmedUp = false;
    let welcomeTimer = null;

    /* ==================================================================
     * 6. RENDERING
     * ================================================================== */

    function appendInlineText(parent, text) {
        text.split(/(\*\*[^*]+\*\*)/g).forEach((part) => {
            if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
                const strong = document.createElement("strong");
                strong.textContent = part.slice(2, -2);
                parent.appendChild(strong);
            } else {
                parent.appendChild(document.createTextNode(part));
            }
        });
    }

    function renderAnswer(parent, text) {
        let currentList = null;

        String(text || "").split(/\r?\n/).forEach((line) => {
            const trimmed = line.trim();

            if (!trimmed) {
                currentList = null;
                return;
            }

            const bullet = trimmed.match(/^[-*]\s+(.+)$/);

            if (bullet) {
                if (!currentList) {
                    currentList = document.createElement("ul");
                    parent.appendChild(currentList);
                }

                const li = document.createElement("li");
                appendInlineText(li, bullet[1]);
                currentList.appendChild(li);
                return;
            }

            currentList = null;

            const p = document.createElement("p");
            appendInlineText(p, trimmed);
            parent.appendChild(p);
        });
    }

    function buildSources(sources) {
        if (!Array.isArray(sources) || !sources.length) return null;

        const seen = new Set();
        const links = [];

        sources.forEach((source) => {
            if (!source || !source.url || seen.has(source.url)) return;
            seen.add(source.url);

            let safeUrl;

            try {
                const parsed = new URL(source.url);
                if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return;
                safeUrl = parsed.href;
            } catch {
                return;
            }

            const link = document.createElement("a");
            link.className = "alcor-source";
            link.href = safeUrl;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.textContent = source.title || source.url;
            links.push(link);
        });

        if (!links.length) return null;

        const box = document.createElement("div");
        box.className = "alcor-sources";

        const title = document.createElement("div");
        title.className = "alcor-sources-title";
        title.textContent = "Sources";

        box.appendChild(title);
        links.forEach((link) => box.appendChild(link));

        return box;
    }

    function scrollToBottom() {
        requestAnimationFrame(() => {
            messages.scrollTop = messages.scrollHeight;
        });
    }

    function addMessage(type, text, sources = []) {
        const wrapper = document.createElement("div");
        wrapper.className = `alcor-message ${type}`;

        const bubble = document.createElement("div");
        bubble.className = "alcor-bubble";

        if (type === "bot") {
            renderAnswer(bubble, text);
            const sourceBox = buildSources(sources);
            if (sourceBox) bubble.appendChild(sourceBox);
        } else {
            bubble.textContent = text;
        }

        wrapper.appendChild(bubble);
        messages.appendChild(wrapper);
        scrollToBottom();

        return wrapper;
    }

    function addTypingIndicator() {
        const wrapper = document.createElement("div");
        wrapper.className = "alcor-message bot";

        const bubble = document.createElement("div");
        bubble.className = "alcor-bubble";

        const typing = document.createElement("div");
        typing.className = "alcor-typing";

        for (let i = 0; i < 3; i++) {
            const dot = document.createElement("span");
            dot.className = "alcor-dot";
            typing.appendChild(dot);
        }

        bubble.appendChild(typing);
        wrapper.appendChild(bubble);
        messages.appendChild(wrapper);
        scrollToBottom();

        let note = null;

        return {
            remove() {
                wrapper.remove();
            },
            setNote(text) {
                if (!note) {
                    note = document.createElement("div");
                    note.className = "alcor-wait-note";
                    bubble.appendChild(note);
                }
                note.textContent = text;
                scrollToBottom();
            }
        };
    }

    function addErrorMessage(question, ownChatId) {
        const wrapper = addMessage(
            "bot",
            "I couldn't reach the assistant. Please check your connection and try again."
        );

        const retry = document.createElement("button");
        retry.type = "button";
        retry.className = "alcor-retry";
        retry.textContent = "Try again";

        retry.addEventListener("click", () => {
            if (requestInProgress || ownChatId !== chatId) return;
            wrapper.remove();
            sendMessage(question, { isRetry: true });
        });

        wrapper.querySelector(".alcor-bubble").appendChild(retry);
    }

    function showWelcomeMessage() {
        const wrapper = document.createElement("div");
        wrapper.className = "alcor-message bot";

        const bubble = document.createElement("div");
        bubble.className = "alcor-bubble";

        const title = document.createElement("p");
        title.className = "alcor-welcome-title";
        title.textContent = CONFIG.welcomeTitle;

        const lead = document.createElement("p");
        lead.className = "alcor-welcome-lead";
        lead.textContent = CONFIG.welcomeLead;

        const list = document.createElement("ul");

        CONFIG.welcomePoints.forEach((point) => {
            const li = document.createElement("li");
            li.textContent = point;
            list.appendChild(li);
        });

        bubble.append(title, lead, list);
        wrapper.appendChild(bubble);
        messages.appendChild(wrapper);
        scrollToBottom();
    }

    /* ==================================================================
     * 7. ROTATING EXAMPLE QUESTIONS
     * ================================================================== */

    function renderHint() {
        hintEl.textContent = "";

        const question = document.createElement("b");
        question.textContent = CONFIG.hints[hintIndex % CONFIG.hints.length];

        hintEl.append(document.createTextNode("Try asking: "), question);
    }

    function stopHints() {
        if (hintTimer) {
            clearInterval(hintTimer);
            hintTimer = null;
        }
    }

    function startHints() {
        stopHints();

        if (input.value || document.activeElement === input) return;

        hintEl.classList.remove("is-hidden", "is-swapping");
        renderHint();

        hintTimer = setInterval(() => {
            hintEl.classList.add("is-swapping");

            setTimeout(() => {
                hintIndex = (hintIndex + 1) % CONFIG.hints.length;
                renderHint();
                hintEl.classList.remove("is-swapping");
            }, 260);
        }, CONFIG.hintIntervalMs);
    }

    function hideHints() {
        stopHints();
        hintEl.classList.add("is-hidden");
    }

    input.addEventListener("focus", hideHints);
    input.addEventListener("input", () => { if (input.value) hideHints(); });
    input.addEventListener("blur", () => { if (isOpen && !input.value) startHints(); });

    /* ==================================================================
     * 8. NETWORK + COLD-START HANDLING
     * ================================================================== */

    // Fire-and-forget request that wakes a sleeping backend.
    function warmUp() {
        if (warmedUp) return;
        warmedUp = true;

        try {
            fetch(HEALTH_URL, { method: "GET", mode: "no-cors", cache: "no-store" })
                .catch(() => {});
        } catch (e) {
            /* ignore */
        }
    }

    const isTransientStatus = (status) =>
        [408, 425, 429, 500, 502, 503, 504].includes(status);

    function looksUnavailable(data) {
        const text = String((data && (data.answer || data.detail || data.error)) || "");
        return /temporarily unavailable|service unavailable|starting up|try again in a moment/i.test(text);
    }

    const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

    // Sends the request and quietly retries while the backend wakes up.
    async function requestAnswer(payload, onSlow) {
        const started = Date.now();
        const slowTimer = setTimeout(onSlow, CONFIG.slowNoticeAfterMs);
        let lastError = null;

        try {
            for (let attempt = 1; attempt <= CONFIG.maxAttempts; attempt++) {
                if (Date.now() - started > CONFIG.totalDeadlineMs) break;

                const controller = new AbortController();
                const timeout = setTimeout(() => controller.abort(), CONFIG.attemptTimeoutMs);

                try {
                    const response = await fetch(API_URL, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload),
                        signal: controller.signal
                    });

                    clearTimeout(timeout);

                    if (response.ok) {
                        let data = null;

                        try {
                            data = await response.json();
                        } catch (e) {
                            data = null;
                        }

                        if (data && !looksUnavailable(data)) return data;

                        lastError = new Error("Backend not ready");
                    } else if (isTransientStatus(response.status)) {
                        lastError = new Error(`HTTP ${response.status}`);
                    } else {
                        // Real client error: retrying will not help.
                        const fatal = new Error(`HTTP ${response.status}`);
                        fatal.fatal = true;
                        throw fatal;
                    }
                } catch (error) {
                    clearTimeout(timeout);
                    if (error && error.fatal) throw error;

                    // Network failure / timeout while the server is starting.
                    lastError = error;
                }

                if (attempt < CONFIG.maxAttempts) {
                    await sleep(Math.min(1000 * Math.pow(1.6, attempt - 1), 6000));
                }
            }
        } finally {
            clearTimeout(slowTimer);
        }

        throw lastError || new Error("Request failed");
    }

    /* ==================================================================
     * 9. ACTIONS
     * ================================================================== */

    async function sendMessage(text, options = {}) {
        const question = (typeof text === "string" ? text : input.value).trim();

        if (!question || requestInProgress) return;

        const ownChatId = chatId;

        if (!options.isRetry) addMessage("user", question);

        input.value = "";
        requestInProgress = true;
        sendButton.disabled = true;
        input.disabled = true;

        const typingMessage = addTypingIndicator();

        try {
            const data = await requestAnswer(
                {
                    message: question,
                    history: conversationHistory.slice(-CONFIG.historySent)
                },
                () => typingMessage.setNote(
                    "Getting things ready. This can take a few seconds."
                )
            );

            typingMessage.remove();

            // "New chat" was clicked while waiting: drop this reply.
            if (ownChatId !== chatId) return;

            const answer =
                data.answer ||
                "I don't have that information in the Alcor knowledge base.";

            conversationHistory.push(
                { role: "user", content: question },
                { role: "assistant", content: answer }
            );

            conversationHistory = conversationHistory.slice(-CONFIG.historyStored);

            addMessage("bot", answer, data.sources || []);

        } catch (error) {
            console.error("Ask Alcor error:", error);

            typingMessage.remove();

            if (ownChatId === chatId) addErrorMessage(question, ownChatId);

        } finally {
            requestInProgress = false;
            sendButton.disabled = false;
            input.disabled = false;

            if (isOpen) input.focus();
        }
    }

    function startNewChat() {
        chatId++;
        conversationHistory = [];
        messages.innerHTML = "";
        clearTimeout(welcomeTimer);

        showWelcomeMessage();

        // Unlock the input if an older request was still pending.
        requestInProgress = false;
        sendButton.disabled = false;
        input.disabled = false;
        input.value = "";

        input.blur();
        startHints();
    }

    function openChat() {
        isOpen = true;
        container.classList.add("is-open");
        launcher.setAttribute("aria-expanded", "true");
        launcher.setAttribute("aria-label", "Close Ask Alcor chat");

        warmUp();

        if (!messages.children.length) {
            welcomeTimer = setTimeout(() => {
                if (!messages.children.length) showWelcomeMessage();
            }, 350);
        }

        scrollToBottom();
        startHints();
    }

    function closeChat() {
        isOpen = false;
        container.classList.remove("is-open");
        launcher.setAttribute("aria-expanded", "false");
        launcher.setAttribute("aria-label", "Open Ask Alcor chat");
        stopHints();
        launcher.focus();
    }

    /* ==================================================================
     * 10. EVENTS
     * ================================================================== */

    launcher.addEventListener("click", () => (isOpen ? closeChat() : openChat()));
    closeButton.addEventListener("click", closeChat);
    newChatButton.addEventListener("click", startNewChat);

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        sendMessage();
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && isOpen) closeChat();
    });

    // Wake the backend early so it is usually ready when someone asks.
    launcher.addEventListener("mouseenter", warmUp);
    launcher.addEventListener("touchstart", warmUp, { passive: true });

    if (document.readyState === "complete") warmUp();
    else window.addEventListener("load", warmUp);

})();