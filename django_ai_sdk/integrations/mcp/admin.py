"""Django admin for integrations.

``MCPServerConfig`` is editable - this is the non-developer, no-deploy way to add,
edit, and enable/disable an MCP server (see ``integrations/registry.py``). Secret
fields render as blank password inputs; leaving one blank on edit preserves the
existing stored secret rather than clearing it.

``MCPOAuthToken``/``MCPOAuthClient`` are read-only - observability into who's
connected to what, without ever rendering token/secret material (even encrypted).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from django import forms
from django.contrib import admin

from django_ai_sdk.integrations.mcp.models import MCPOAuthClient, MCPOAuthToken, MCPServerConfig

if TYPE_CHECKING:
    from django.http import HttpRequest


class MCPServerConfigForm(forms.ModelForm):
    token = forms.CharField(
        required=False,
        widget=forms.PasswordInput(render_value=False),
        help_text="Leave blank to keep the existing token unchanged. Only used when auth='token'.",
    )
    client_secret = forms.CharField(
        required=False,
        widget=forms.PasswordInput(render_value=False),
        help_text="Leave blank to keep the existing client secret unchanged.",
    )

    class Meta:
        model = MCPServerConfig
        fields = "__all__"

    def save(self, commit: bool = True) -> MCPServerConfig:
        instance = super().save(commit=False)
        if token := self.cleaned_data.get("token"):
            instance.set_token(token)
        if client_secret := self.cleaned_data.get("client_secret"):
            instance.set_client_secret(client_secret)
        if commit:
            instance.save()
        return instance


@admin.register(MCPServerConfig)
class MCPServerConfigAdmin(admin.ModelAdmin):
    form = MCPServerConfigForm
    list_display = ("name", "label", "auth", "url", "enabled", "updated_at")
    list_filter = ("auth", "enabled")
    search_fields = ("name", "label", "url")
    readonly_fields = ("created_at", "updated_at")
    fieldsets = (
        (None, {"fields": ("name", "label", "hint", "url", "enabled")}),
        ("Auth", {"fields": ("auth", "token", "client_id", "client_secret", "scope")}),
        (
            "OAuth discovery overrides (optional — auto-discovered if left blank)",
            {
                "classes": ("collapse",),
                "fields": ("oauth_discovery_url", "authorization_endpoint", "token_endpoint"),
            },
        ),
        ("Tools", {"fields": ("tools",)}),
        ("Metadata", {"fields": ("created_at", "updated_at")}),
    )


@admin.register(MCPOAuthToken)
class MCPOAuthTokenAdmin(admin.ModelAdmin):
    """Read-only — observability only. Token material is never rendered, even encrypted."""

    list_display = ("user", "server_name", "is_expired_display", "expires_at")
    list_filter = ("server_name",)
    search_fields = ("server_name",)
    fields = ("user", "server_name", "token_type", "expires_at", "scope", "is_expired_display")
    readonly_fields = fields

    def has_add_permission(self, request: HttpRequest) -> bool:
        return False

    @admin.display(description="Expired", boolean=True)
    def is_expired_display(self, obj: MCPOAuthToken) -> bool:
        return obj.is_expired()


@admin.register(MCPOAuthClient)
class MCPOAuthClientAdmin(admin.ModelAdmin):
    """Read-only — observability only. ``client_secret`` is never rendered."""

    list_display = ("server_name", "client_id", "registered_at")
    search_fields = ("server_name", "client_id")
    fields = ("server_name", "client_id", "redirect_uri", "registered_at")
    readonly_fields = fields

    def has_add_permission(self, request: HttpRequest) -> bool:
        return False
