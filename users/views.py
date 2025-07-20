from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.views import LoginView
from django.urls import reverse_lazy
from .forms import UserRegistrationForm, UserLoginForm, UserProfileForm

def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Account created successfully! Welcome to Research Paper Analyzer.')
            return redirect('users:dashboard')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = UserRegistrationForm()
    
    return render(request, 'users/register.html', {'form': form})

class CustomLoginView(LoginView):
    form_class = UserLoginForm
    template_name = 'users/login.html'
    success_url = reverse_lazy('users:dashboard')
    
    def get_success_url(self):
        return self.success_url
    
    def form_valid(self, form):
        remember_me = form.cleaned_data.get('remember_me')
        if not remember_me:
            self.request.session.set_expiry(0)
        return super().form_valid(form)

@login_required
def user_logout(request):
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('users:login')

@login_required
def profile(request):
    if request.method == 'POST':
        form = UserProfileForm(request.POST, request.FILES, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, 'Profile updated successfully!')
            return redirect('users:profile')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = UserProfileForm(instance=request.user)
    
    context = {
        'form': form,
        'user': request.user
    }
    return render(request, 'users/profile.html', context)

@login_required
def dashboard(request):
    """User dashboard showing recent papers and analysis status"""
    user_papers = request.user.paper_set.all()[:10]  # Get last 10 papers
    recent_analyses = []
    
    for paper in user_papers:
        if hasattr(paper, 'analysis'):
            recent_analyses.append(paper.analysis)
    
    context = {
        'user_papers': user_papers,
        'recent_analyses': recent_analyses,
        'total_papers': user_papers.count(),
        'completed_analyses': len([p for p in user_papers if p.status == 'completed']),
        'pending_analyses': len([p for p in user_papers if p.status == 'pending']),
    }
    return render(request, 'users/dashboard.html', context)
