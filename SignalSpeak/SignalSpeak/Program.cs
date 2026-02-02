using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Authorization;
using Microsoft.AspNetCore.Components.Server;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using SignalSpeak.Components;
using SignalSpeak.Data;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

// ✅ HttpClient para poder hacer POST desde Blazor
builder.Services.AddHttpClient();

builder.Services.AddScoped(sp =>
{
    var nav = sp.GetRequiredService<NavigationManager>();
    return new HttpClient { BaseAddress = new Uri(nav.BaseUri) };
});
// 🔒 RUTA ABSOLUTA A LA DB (NO relativa)
var dbFolder = Path.Combine(builder.Environment.ContentRootPath, "App_Data");
Directory.CreateDirectory(dbFolder);

var dbPath = Path.Combine(dbFolder, "SignSpeakAuth.db");
Console.WriteLine("👉 SQLITE DB PATH: " + dbPath);

builder.Services.AddDbContext<ApplicationDbContext>(options =>
    options.UseSqlite($"Data Source={dbPath}"));

// Identity
builder.Services.AddIdentityCore<IdentityUser>(options =>
{
    options.Password.RequiredLength = 6;
    options.Password.RequireDigit = false;
    options.Password.RequireNonAlphanumeric = false;
    options.Password.RequireUppercase = false;
    options.Password.RequireLowercase = false;
})
.AddEntityFrameworkStores<ApplicationDbContext>()
.AddSignInManager()
.AddDefaultTokenProviders();

// Auth cookies (Identity)
builder.Services.AddAuthentication(IdentityConstants.ApplicationScheme)
    .AddIdentityCookies();

builder.Services.AddAuthorization();
builder.Services.AddCascadingAuthenticationState();
builder.Services.AddScoped<AuthenticationStateProvider, ServerAuthenticationStateProvider>();

var app = builder.Build();

// ✅ Crea / actualiza tablas con migraciones (mejor que EnsureCreated)
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();
    db.Database.Migrate();
}

app.UseHttpsRedirection();
app.UseStaticFiles();
app.UseRouting();

app.UseAuthentication();
app.UseAuthorization();

app.UseAntiforgery();


// =====================================================
// ✅ AQUÍ SE CREA LA RUTA HTTP: POST /auth/login
// NO ES UNA CARPETA. NO ES UN ARCHIVO.
// ES UNA "RUTA" EN TU SERVIDOR.
// =====================================================
app.MapPost("/auth/login", async (
    LoginDto dto,
    UserManager<IdentityUser> userManager,
    SignInManager<IdentityUser> signInManager) =>
{
    var email = (dto.Email ?? "").Trim();

    var user = await userManager.FindByEmailAsync(email);
    if (user is null)
        return Results.Unauthorized();

    var result = await signInManager.PasswordSignInAsync(
        user.UserName!,
        dto.Password ?? "",
        dto.RememberMe,
        lockoutOnFailure: false);

    return result.Succeeded ? Results.Ok() : Results.Unauthorized();
})
.DisableAntiforgery(); // evita 400 por antiforgery en dev


app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();

// DTO del endpoint
public record LoginDto(string? Email, string? Password, bool RememberMe);
