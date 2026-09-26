(function () {
    "use strict";

    const API_URL = "http://127.0.0.1:8000/api/chat";

    const container = document.createElement("div");
    container.id = "ask-alcor-widget";

    container.innerHTML = `
        <style>
            #ask-alcor-widget,
            #ask-alcor-widget * {
                box-sizing: border-box;
            }

            #ask-alcor-widget {
                font-family:
                    -apple-system,
                    BlinkMacSystemFont,
                    "Segoe UI",
                    Roboto,
                    Arial,
                    sans-serif;
            }

            /* -----------------------------
               Launcher
            ----------------------------- */

            #ask-alcor-widget .alcor-launcher {
                position: fixed;
                right: 24px;
                bottom: 24px;

                width: 60px;
                height: 60px;

                border: none;
                border-radius: 50%;

                background: #111827;
                color: #ffffff;

                font-size: 25px;

                cursor: pointer;

                box-shadow:
                    0 8px 25px rgba(0, 0, 0, 0.20);

                z-index: 999999;
            }

            #ask-alcor-widget .alcor-launcher:hover {
                transform: translateY(-2px);
            }

            /* -----------------------------
               Chat window
            ----------------------------- */

            #ask-alcor-widget .alcor-chat {
                position: fixed;

                right: 24px;
                bottom: 96px;

                width: 390px;
                max-width: calc(100vw - 32px);

                height: 600px;
                max-height: calc(100vh - 120px);

                background: #ffffff;

                border: 1px solid #e5e7eb;
                border-radius: 20px;

                box-shadow:
                    0 20px 60px rgba(0, 0, 0, 0.20);

                overflow: hidden;

                display: none;
                flex-direction: column;

                z-index: 999999;
            }

            /* -----------------------------
               Header
            ----------------------------- */

            #ask-alcor-widget .alcor-header {
                background: #111827;
                color: #ffffff;

                padding: 17px 18px;

                display: flex;
                align-items: center;
                justify-content: space-between;

                flex-shrink: 0;
            }

            #ask-alcor-widget .alcor-brand {
                display: flex;
                align-items: center;
                gap: 11px;
            }

            #ask-alcor-widget .alcor-logo {
                width: 40px;
                height: 40px;

                border-radius: 50%;

                background: #ffffff;
                color: #111827;

                display: flex;
                align-items: center;
                justify-content: center;

                font-size: 19px;
                font-weight: 700;
            }

            #ask-alcor-widget .alcor-title {
                font-size: 17px;
                font-weight: 700;
            }

            #ask-alcor-widget .alcor-status {
                margin-top: 2px;

                font-size: 12px;

                color: #d1d5db;
            }

            #ask-alcor-widget .alcor-header-actions {
                display: flex;
                align-items: center;
                gap: 4px;
            }

            #ask-alcor-widget .alcor-new-chat,
            #ask-alcor-widget .alcor-close {
                border: none;
                background: transparent;
                color: #ffffff;

                cursor: pointer;

                border-radius: 8px;
            }

            #ask-alcor-widget .alcor-new-chat {
                padding: 7px 9px;
                font-size: 12px;
            }

            #ask-alcor-widget .alcor-new-chat:hover {
                background: rgba(255, 255, 255, 0.10);
            }

            #ask-alcor-widget .alcor-close {
                width: 34px;
                height: 34px;

                font-size: 25px;
            }

            #ask-alcor-widget .alcor-close:hover {
                background: rgba(255, 255, 255, 0.10);
            }

            /* -----------------------------
               Messages
            ----------------------------- */

            #ask-alcor-widget .alcor-messages {
                flex: 1;

                overflow-y: auto;

                padding: 18px;

                background: #f8fafc;

                scroll-behavior: smooth;
            }

            #ask-alcor-widget .alcor-message {
                display: flex;

                margin-bottom: 15px;
            }

            #ask-alcor-widget .alcor-message.bot {
                justify-content: flex-start;
            }

            #ask-alcor-widget .alcor-message.user {
                justify-content: flex-end;
            }

            #ask-alcor-widget .alcor-bubble {
                max-width: 84%;

                padding: 12px 14px;

                border-radius: 15px;

                font-size: 14px;
                line-height: 1.55;

                word-wrap: break-word;
                overflow-wrap: anywhere;
            }

            #ask-alcor-widget .bot .alcor-bubble {
                background: #ffffff;

                color: #1f2937;

                border: 1px solid #e5e7eb;

                border-bottom-left-radius: 5px;
            }

            #ask-alcor-widget .user .alcor-bubble {
                background: #111827;

                color: #ffffff;

                border-bottom-right-radius: 5px;
            }

            /* -----------------------------
               Markdown-like answer formatting
            ----------------------------- */

            #ask-alcor-widget .alcor-bubble p {
                margin: 0 0 8px;
            }

            #ask-alcor-widget .alcor-bubble p:last-child {
                margin-bottom: 0;
            }

            #ask-alcor-widget .alcor-bubble strong {
                font-weight: 700;
            }

            #ask-alcor-widget .alcor-bubble ul {
                margin: 7px 0 4px;
                padding-left: 20px;
            }

            #ask-alcor-widget .alcor-bubble li {
                margin-bottom: 4px;
            }

            /* -----------------------------
               Sources
            ----------------------------- */

            #ask-alcor-widget .alcor-sources {
                margin-top: 12px;

                padding-top: 9px;

                border-top: 1px solid #e5e7eb;
            }

            #ask-alcor-widget .alcor-sources-title {
                font-size: 11px;
                font-weight: 700;

                color: #6b7280;

                margin-bottom: 5px;
            }

            #ask-alcor-widget .alcor-source {
                display: block;

                margin-top: 4px;

                color: #2563eb;

                font-size: 11px;

                text-decoration: none;
            }

            #ask-alcor-widget .alcor-source:hover {
                text-decoration: underline;
            }

            /* -----------------------------
               Suggestions
            ----------------------------- */

            #ask-alcor-widget .alcor-suggestions {
                display: flex;

                flex-wrap: wrap;

                gap: 7px;

                margin-top: 12px;
            }

            #ask-alcor-widget .alcor-suggestion {
                border: 1px solid #d1d5db;

                background: #ffffff;

                color: #374151;

                border-radius: 18px;

                padding: 7px 11px;

                font-size: 12px;

                cursor: pointer;
            }

            #ask-alcor-widget .alcor-suggestion:hover {
                background: #f3f4f6;
            }

            /* -----------------------------
               Typing indicator
            ----------------------------- */

            #ask-alcor-widget .alcor-typing {
                display: flex;

                align-items: center;

                gap: 4px;

                height: 18px;
            }

            #ask-alcor-widget .alcor-dot {
                width: 6px;
                height: 6px;

                border-radius: 50%;

                background: #9ca3af;

                animation: alcorTyping 1.2s infinite;
            }

            #ask-alcor-widget .alcor-dot:nth-child(2) {
                animation-delay: 0.15s;
            }

            #ask-alcor-widget .alcor-dot:nth-child(3) {
                animation-delay: 0.30s;
            }

            @keyframes alcorTyping {
                0%,
                60%,
                100% {
                    transform: translateY(0);
                    opacity: 0.45;
                }

                30% {
                    transform: translateY(-4px);
                    opacity: 1;
                }
            }

            /* -----------------------------
               Input
            ----------------------------- */

            #ask-alcor-widget .alcor-input-area {
                display: flex;

                gap: 8px;

                padding: 12px;

                background: #ffffff;

                border-top: 1px solid #e5e7eb;

                flex-shrink: 0;
            }

            #ask-alcor-widget .alcor-input {
                flex: 1;

                min-width: 0;

                border: 1px solid #d1d5db;

                border-radius: 11px;

                padding: 11px 13px;

                font-size: 14px;

                outline: none;

                color: #111827;
            }

            #ask-alcor-widget .alcor-input:focus {
                border-color: #111827;
            }

            #ask-alcor-widget .alcor-input::placeholder {
                color: #9ca3af;
            }

            #ask-alcor-widget .alcor-send {
                border: none;

                border-radius: 11px;

                background: #111827;

                color: #ffffff;

                padding: 0 16px;

                font-size: 14px;

                font-weight: 600;

                cursor: pointer;
            }

            #ask-alcor-widget .alcor-send:hover:not(:disabled) {
                background: #1f2937;
            }

            #ask-alcor-widget .alcor-send:disabled {
                opacity: 0.55;

                cursor: not-allowed;
            }

            /* -----------------------------
               Mobile
            ----------------------------- */

            @media (max-width: 480px) {
                #ask-alcor-widget .alcor-chat {
                    right: 8px;
                    bottom: 78px;

                    width: calc(100vw - 16px);

                    height: calc(100vh - 95px);

                    max-height: none;
                }

                #ask-alcor-widget .alcor-launcher {
                    right: 16px;
                    bottom: 16px;
                }
            }
        </style>

        <button
            class="alcor-launcher"
            aria-label="Open Ask Alcor"
            title="Ask Alcor"
        >
            💬
        </button>

        <section
            class="alcor-chat"
            aria-label="Ask Alcor chatbot"
        >

            <header class="alcor-header">

                <div class="alcor-brand">

                    <div class="alcor-logo">
                        A
                    </div>

                    <div>
                        <div class="alcor-title">
                            Ask Alcor
                        </div>

                        <div class="alcor-status">
                            Alcor Solutions AI Assistant
                        </div>
                    </div>

                </div>

                <div class="alcor-header-actions">

                    <button
                        class="alcor-new-chat"
                        type="button"
                        title="Start a new conversation"
                    >
                        New chat
                    </button>

                    <button
                        class="alcor-close"
                        type="button"
                        aria-label="Close chat"
                    >
                        ×
                    </button>

                </div>

            </header>

            <div class="alcor-messages"></div>

            <form class="alcor-input-area">

                <input
                    class="alcor-input"
                    type="text"
                    placeholder="Ask me about Alcor..."
                    autocomplete="off"
                    aria-label="Message"
                />

                <button
                    class="alcor-send"
                    type="submit"
                >
                    Send
                </button>

            </form>

        </section>
    `;

    document.body.appendChild(container);

    const launcher = container.querySelector(".alcor-launcher");
    const chat = container.querySelector(".alcor-chat");
    const closeButton = container.querySelector(".alcor-close");
    const newChatButton = container.querySelector(".alcor-new-chat");
    const messages = container.querySelector(".alcor-messages");
    const form = container.querySelector(".alcor-input-area");
    const input = container.querySelector(".alcor-input");
    const sendButton = container.querySelector(".alcor-send");

    /*
     * Conversation state.
     *
     * This is intentionally generic.
     * The model decides what "he", "she", "it", "that",
     * "yes", "tell me more", etc. refer to.
     */
    let conversationHistory = [];

    let requestInProgress = false;

    /*
     * -----------------------------
     * Safe text / Markdown rendering
     * -----------------------------
     */

    function appendInlineText(parent, text) {
        const parts = text.split(/(\*\*[^*]+\*\*)/g);

        parts.forEach((part) => {

            if (
                part.startsWith("**") &&
                part.endsWith("**")
            ) {
                const strong = document.createElement("strong");

                strong.textContent = part.slice(2, -2);

                parent.appendChild(strong);

            } else {
                parent.appendChild(
                    document.createTextNode(part)
                );
            }
        });
    }


    function renderAnswer(parent, text) {

        const lines = String(text || "").split(/\r?\n/);

        let currentList = null;

        function closeList() {
            currentList = null;
        }

        lines.forEach((line) => {

            const trimmed = line.trim();

            if (!trimmed) {
                closeList();
                return;
            }

            /*
             * Bullet list
             */
            const bulletMatch = trimmed.match(
                /^[-*]\s+(.+)$/
            );

            if (bulletMatch) {

                if (!currentList) {
                    currentList = document.createElement("ul");
                    parent.appendChild(currentList);
                }

                const li = document.createElement("li");

                appendInlineText(
                    li,
                    bulletMatch[1]
                );

                currentList.appendChild(li);

                return;
            }

            closeList();

            /*
             * Normal paragraph
             */
            const paragraph = document.createElement("p");

            appendInlineText(
                paragraph,
                trimmed
            );

            parent.appendChild(paragraph);
        });
    }


    /*
     * -----------------------------
     * Add message
     * -----------------------------
     */

    function addMessage(
        type,
        text,
        sources = []
    ) {

        const wrapper = document.createElement("div");

        wrapper.className =
            `alcor-message ${type}`;


        const bubble = document.createElement("div");

        bubble.className = "alcor-bubble";


        if (type === "bot") {

            renderAnswer(
                bubble,
                text
            );

        } else {

            bubble.textContent = text;

        }


        wrapper.appendChild(bubble);


        /*
         * Sources
         */

        if (
            type === "bot" &&
            Array.isArray(sources) &&
            sources.length > 0
        ) {

            const uniqueSources = [];

            const seen = new Set();

            sources.forEach((source) => {

                if (
                    !source ||
                    !source.url ||
                    seen.has(source.url)
                ) {
                    return;
                }

                seen.add(source.url);

                uniqueSources.push(source);

            });


            if (uniqueSources.length > 0) {

                const sourceBox =
                    document.createElement("div");

                sourceBox.className =
                    "alcor-sources";


                const title =
                    document.createElement("div");

                title.className =
                    "alcor-sources-title";

                title.textContent =
                    "Sources";

                sourceBox.appendChild(title);


                uniqueSources.forEach((source) => {

                    /*
                     * Only allow normal web URLs.
                     */
                    let safeUrl;

                    try {

                        const parsed =
                            new URL(source.url);

                        if (
                            parsed.protocol !== "http:" &&
                            parsed.protocol !== "https:"
                        ) {
                            return;
                        }

                        safeUrl = parsed.href;

                    } catch {
                        return;
                    }


                    const link =
                        document.createElement("a");

                    link.className =
                        "alcor-source";

                    link.href = safeUrl;

                    link.target = "_blank";

                    link.rel =
                        "noopener noreferrer";

                    link.textContent =
                        source.title || source.url;

                    sourceBox.appendChild(link);
                });


                bubble.appendChild(sourceBox);
            }
        }


        messages.appendChild(wrapper);

        scrollToBottom();

        return wrapper;
    }


    /*
     * -----------------------------
     * Typing indicator
     * -----------------------------
     */

    function addTypingIndicator() {

        const wrapper =
            document.createElement("div");

        wrapper.className =
            "alcor-message bot";


        const bubble =
            document.createElement("div");

        bubble.className =
            "alcor-bubble";


        const typing =
            document.createElement("div");

        typing.className =
            "alcor-typing";


        for (let i = 0; i < 3; i++) {

            const dot =
                document.createElement("span");

            dot.className =
                "alcor-dot";

            typing.appendChild(dot);
        }


        bubble.appendChild(typing);

        wrapper.appendChild(bubble);

        messages.appendChild(wrapper);

        scrollToBottom();

        return wrapper;
    }


    /*
     * -----------------------------
     * Scroll
     * -----------------------------
     */

    function scrollToBottom() {

        requestAnimationFrame(() => {

            messages.scrollTop =
                messages.scrollHeight;

        });
    }


    /*
     * -----------------------------
     * Welcome message
     * -----------------------------
     */

    function showWelcomeMessage() {

        const wrapper = addMessage(
            "bot",
            "Hi! 👋 I'm Ask Alcor, the AI assistant for Alcor Solutions. How can I help you today?"
        );


        const suggestions =
            document.createElement("div");

        suggestions.className =
            "alcor-suggestions";


        const questions = [
            "Who is the CEO of Alcor?",
            "What does Alcor do?",
            "Tell me about Alcor's leadership"
        ];


        questions.forEach((question) => {

            const button =
                document.createElement("button");

            button.type = "button";

            button.className =
                "alcor-suggestion";

            button.textContent =
                question;


            button.addEventListener(
                "click",
                () => {

                    input.value =
                        question;

                    sendMessage();

                }
            );


            suggestions.appendChild(button);
        });


        wrapper
            .querySelector(".alcor-bubble")
            .appendChild(suggestions);
    }


    /*
     * -----------------------------
     * New conversation
     * -----------------------------
     */

    function startNewChat() {

        conversationHistory = [];

        messages.innerHTML = "";

        showWelcomeMessage();

        input.focus();
    }


    /*
     * -----------------------------
     * Send message
     * -----------------------------
     */

    async function sendMessage() {

        const question =
            input.value.trim();


        if (
            !question ||
            requestInProgress
        ) {
            return;
        }


        /*
         * Display user message immediately.
         */

        addMessage(
            "user",
            question
        );


        input.value = "";

        requestInProgress = true;

        sendButton.disabled = true;

        input.disabled = true;


        /*
         * Show live typing state.
         */

        const typingMessage =
            addTypingIndicator();


        try {

            const response =
                await fetch(
                    API_URL,
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({

                            message: question,

                            /*
                             * This is the important part:
                             * send previous conversation turns.
                             */

                            history:
                                conversationHistory

                        })
                    }
                );


            if (!response.ok) {

                throw new Error(
                    `HTTP ${response.status}`
                );
            }


            const data =
                await response.json();


            typingMessage.remove();


            const answer =
                data.answer ||
                "I don't have that information in the Alcor knowledge base.";


            /*
             * Save both sides of the conversation
             * only after the backend successfully responds.
             */

            conversationHistory.push(
                {
                    role: "user",
                    content: question
                },
                {
                    role: "assistant",
                    content: answer
                }
            );


            /*
             * Keep browser memory bounded.
             */

            if (
                conversationHistory.length > 20
            ) {

                conversationHistory =
                    conversationHistory.slice(-20);

            }


            addMessage(
                "bot",
                answer,
                data.sources || []
            );


        } catch (error) {

            console.error(
                "Ask Alcor error:",
                error
            );


            typingMessage.remove();


            addMessage(
                "bot",
                "I'm having trouble connecting right now. Please try again in a moment."
            );


        } finally {

            requestInProgress = false;

            sendButton.disabled = false;

            input.disabled = false;

            input.focus();

        }
    }


    /*
     * -----------------------------
     * Open / close
     * -----------------------------
     */

    launcher.addEventListener(
        "click",
        () => {

            chat.style.display =
                "flex";


            if (
                !messages.children.length
            ) {
                showWelcomeMessage();
            }


            input.focus();

            scrollToBottom();
        }
    );


    closeButton.addEventListener(
        "click",
        () => {

            chat.style.display =
                "none";

        }
    );


    newChatButton.addEventListener(
        "click",
        () => {

            startNewChat();

        }
    );


    /*
     * -----------------------------
     * Submit
     * -----------------------------
     */

    form.addEventListener(
        "submit",
        (event) => {

            event.preventDefault();

            sendMessage();

        }
    );

})();