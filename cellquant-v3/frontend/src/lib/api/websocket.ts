/**
 * WebSocket client for real-time progress updates.
 * Connects to WS /api/v1/ws/{session_id}.
 */

import type { ProgressMessage } from './types';

type MessageHandler = (msg: ProgressMessage) => void;

export class ProgressSocket {
	private ws: WebSocket | null = null;
	private handlers = new Set<MessageHandler>();
	private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
	private sessionId: string;
	private maxRetries = 5;
	private retryCount = 0;

	constructor(sessionId: string) {
		this.sessionId = sessionId;
	}

	connect(): void {
		if (this.ws?.readyState === WebSocket.OPEN) return;

		const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
		const url = `${protocol}//${window.location.host}/api/v1/ws/${this.sessionId}`;

		this.ws = new WebSocket(url);

		this.ws.onopen = () => {
			this.retryCount = 0;
		};

		this.ws.onmessage = (event) => {
			try {
				const msg: ProgressMessage = JSON.parse(event.data);
				if (msg.type === 'heartbeat') return;
				this.handlers.forEach((h) => h(msg));
			} catch {
				// Ignore malformed messages
			}
		};

		this.ws.onclose = () => {
			if (this.retryCount < this.maxRetries) {
				const delay = Math.min(1000 * Math.pow(2, this.retryCount), 10000);
				this.reconnectTimer = setTimeout(() => {
					this.retryCount++;
					this.connect();
				}, delay);
			}
		};

		this.ws.onerror = () => {
			this.ws?.close();
		};
	}

	onMessage(handler: MessageHandler): () => void {
		this.handlers.add(handler);
		return () => this.handlers.delete(handler);
	}

	disconnect(): void {
		if (this.reconnectTimer) {
			clearTimeout(this.reconnectTimer);
			this.reconnectTimer = null;
		}
		this.ws?.close();
		this.ws = null;
		this.handlers.clear();
	}
}
