using System.Linq;
using Microsoft.AspNetCore.Identity;

namespace SignalSpeak.Data;

public class ApplicationUser : IdentityUser
{
    public string? FirstName { get; set; }
    public string? LastName { get; set; }

    // ✅ store the saved image url/path (ex: "/profile-images/USERID.jpg")
    public string? ProfileImagePath { get; set; }

    public string DisplayName =>
        string.Join(" ", new[] { FirstName, LastName }
            .Where(s => !string.IsNullOrWhiteSpace(s)));
}