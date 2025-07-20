from django.contrib import admin
from .models import Paper

@admin.register(Paper)
class PaperAdmin(admin.ModelAdmin):
    list_display = ('title', 'user', 'status', 'created_at', 'file_size_display')
    list_filter = ('status', 'created_at', 'year')
    search_fields = ('title', 'authors', 'abstract')
    readonly_fields = ('file_size', 'processing_time', 'created_at', 'updated_at')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('user', 'title', 'authors', 'year', 'file')
        }),
        ('Metadata', {
            'fields': ('abstract', 'keywords', 'doi', 'journal')
        }),
        ('Analysis Status', {
            'fields': ('status', 'processing_time')
        }),
        ('File Information', {
            'fields': ('file_size', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def file_size_display(self, obj):
        return f"{obj.file_size / (1024*1024):.1f} MB"
    file_size_display.short_description = 'File Size'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')
