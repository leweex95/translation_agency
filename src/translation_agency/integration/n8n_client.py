"""n8n integration utilities for workflow automation."""

import requests
from typing import Dict, Any, Optional
import json
from pathlib import Path

class N8nClient:
    """Client for interacting with n8n workflows."""

    def __init__(self, base_url: str = "http://localhost:5678", api_key: Optional[str] = None):
        """
        Initialize n8n client.

        Args:
            base_url: n8n instance URL
            api_key: n8n API key (if authentication is required)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()

        if api_key:
            self.session.headers.update({"X-N8N-API-KEY": api_key})

    def trigger_workflow(self, workflow_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trigger an n8n workflow via webhook.

        Args:
            workflow_id: Workflow ID or webhook path
            data: Data to send to the workflow

        Returns:
            Workflow execution result
        """
        url = f"{self.base_url}/webhook/{workflow_id}"

        try:
            response = self.session.post(
                url,
                json=data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"n8n workflow trigger failed: {e}")

    def get_workflow_status(self, execution_id: str) -> Dict[str, Any]:
        """
        Get status of a workflow execution.

        Args:
            execution_id: Execution ID

        Returns:
            Execution status
        """
        url = f"{self.base_url}/executions/{execution_id}"

        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to get workflow status: {e}")

class N8nIntegration:
    """Integration utilities for n8n workflows."""

    def __init__(self, workflow_path: str = "n8n/workflow.json"):
        """
        Initialize n8n integration.

        Args:
            workflow_path: Path to n8n workflow JSON file
        """
        self.workflow_path = Path(workflow_path)
        self.workflow_data = None

        if self.workflow_path.exists():
            with open(self.workflow_path, 'r', encoding='utf-8') as f:
                self.workflow_data = json.load(f)

    def get_workflow_info(self) -> Dict[str, Any]:
        """
        Get information about the configured workflow.

        Returns:
            Workflow metadata
        """
        if not self.workflow_data:
            raise FileNotFoundError(f"Workflow file not found: {self.workflow_path}")

        return {
            "name": self.workflow_data.get("name"),
            "nodes": len(self.workflow_data.get("nodes", [])),
            "connections": len(self.workflow_data.get("connections", {})),
            "tags": self.workflow_data.get("tags", [])
        }

    def validate_workflow(self) -> bool:
        """
        Validate that the workflow is properly configured.

        Returns:
            True if workflow is valid
        """
        if not self.workflow_data:
            return False

        required_fields = ["name", "nodes", "connections"]
        for field in required_fields:
            if field not in self.workflow_data:
                return False

        # Check for translation pipeline node
        nodes = self.workflow_data.get("nodes", [])
        has_translation_node = any(
            node.get("name") == "Run Translation Pipeline"
            for node in nodes
        )

        return has_translation_node

    def create_webhook_payload(self, input_document: str, **kwargs) -> Dict[str, Any]:
        """
        Create a webhook payload for the translation workflow.

        Args:
            input_document: Path to input document
            **kwargs: Additional parameters

        Returns:
            Webhook payload
        """
        payload = {
            "input_document": input_document,
            "translation_style": kwargs.get("translation_style", "professional"),
            "target_language": kwargs.get("target_language", "hungarian"),
            "source_language": kwargs.get("source_language", "auto"),
            "output_dir": kwargs.get("output_dir", "output/n8n"),
            "llm_backend": kwargs.get("llm_backend", "chatgpt"),
            "headless": kwargs.get("headless", True),
            "debug": kwargs.get("debug", False)
        }

        if "disabled_steps" in kwargs:
            payload["disabled_steps"] = kwargs["disabled_steps"]

        return payload

# Global instance for easy access
n8n_integration = N8nIntegration()