using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using SignalSpeak.Components;
using SignalSpeak.Data;

var builder = WebApplication.CreateBuilder(args);

// 🔹 Database
builder.Services.AddDbContext<ApplicationDbContext>(options =>
    options.UseSqlServer(
        builder.Configuration.GetConnectionString("DefaultConnection")));

// 🔹 Identity (SINGLE registration)
builder.Services.AddDefaultIdentity<IdentityUser>(options =>
{
    // Sign-in
    options.SignIn.RequireConfirmedAccount = false;

    // Password rules (relaxed for capstone/dev)
    options.Password.RequiredLength = 6;
    options.Password.RequireDigit = false;
    options.Password.RequireLowercase = false;
    options.Password.RequireUppercase = false;
    options.Password.RequireNonAlphanumeric = false;
})
.AddEntityFrameworkStores<ApplicationDbContext>();

// 🔹 Authorization & Auth state
builder.Services.AddAuthorization();
builder.Services.AddCascadingAuthenticationState();

// 🔹 Blazor
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

builder.Services.AddHttpClient();

var app = builder.Build();

// 🔹 Middleware
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();

app.UseRouting();

app.UseAuthentication();
app.UseAuthorization();

app.UseAntiforgery();

// 🔹 Endpoints
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.MapRazorPages();

app.Run();
