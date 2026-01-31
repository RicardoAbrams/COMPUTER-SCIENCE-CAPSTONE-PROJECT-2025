using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;

namespace SignalSpeak.Areas.Identity.Pages.Account
{
    public class LogoutModel : PageModel
    {
        private readonly SignInManager<IdentityUser> _signInManager;

        public LogoutModel(SignInManager<IdentityUser> signInManager)
        {
            _signInManager = signInManager;
        }

        public void OnGet()
        {
            // Show the confirmation page
        }

        public async Task<IActionResult> OnPost()
        {
            await _signInManager.SignOutAsync();
            return LocalRedirect("~/Identity/Account/LogoutSuccess");
        }
    }
}
