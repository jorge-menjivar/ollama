package cmd

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/olekukonko/tablewriter"
	"github.com/spf13/cobra"
	"golang.org/x/crypto/ssh"
	"golang.org/x/term"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/format"
	"github.com/jmorganca/ollama/parser"
	"github.com/jmorganca/ollama/progress"
	"github.com/jmorganca/ollama/readline"
	"github.com/jmorganca/ollama/server"
	"github.com/jmorganca/ollama/version"
)

func CreateHandler(cmd *cobra.Command, args []string) error {
	filename, _ := cmd.Flags().GetString("file")
	filename, err := filepath.Abs(filename)
	if err != nil {
		return err
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	bars := make(map[string]*progress.Bar)

	modelfile, err := os.ReadFile(filename)
	if err != nil {
		return err
	}

	commands, err := parser.Parse(bytes.NewReader(modelfile))
	if err != nil {
		return err
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	status := "transferring model data"
	spinner := progress.NewSpinner(status)
	p.Add(status, spinner)

	for _, c := range commands {
		switch c.Name {
		case "model", "adapter":
			path := c.Args
			if path == "~" {
				path = home
			} else if strings.HasPrefix(path, "~/") {
				path = filepath.Join(home, path[2:])
			}

			if !filepath.IsAbs(path) {
				path = filepath.Join(filepath.Dir(filename), path)
			}

			bin, err := os.Open(path)
			if errors.Is(err, os.ErrNotExist) && c.Name == "model" {
				continue
			} else if err != nil {
				return err
			}
			defer bin.Close()

			hash := sha256.New()
			if _, err := io.Copy(hash, bin); err != nil {
				return err
			}
			bin.Seek(0, io.SeekStart)

			digest := fmt.Sprintf("sha256:%x", hash.Sum(nil))
			if err = client.CreateBlob(cmd.Context(), digest, bin); err != nil {
				return err
			}

			modelfile = bytes.ReplaceAll(modelfile, []byte(c.Args), []byte("@"+digest))
		}
	}

	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			spinner.Stop()

			bar, ok := bars[resp.Digest]
			if !ok {
				bar = progress.NewBar(fmt.Sprintf("pulling %s...", resp.Digest[7:19]), resp.Total, resp.Completed)
				bars[resp.Digest] = bar
				p.Add(resp.Digest, bar)
			}

			bar.Set(resp.Completed)
		} else if status != resp.Status {
			spinner.Stop()

			status = resp.Status
			spinner = progress.NewSpinner(status)
			p.Add(status, spinner)
		}

		return nil
	}

	request := api.CreateRequest{Name: args[0], Modelfile: string(modelfile)}
	if err := client.Create(context.Background(), &request, fn); err != nil {
		return err
	}

	return nil
}

func RunHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	name := args[0]
	// check if the model exists on the server
	_, err = client.Show(context.Background(), &api.ShowRequest{Name: name})
	var statusError api.StatusError
	switch {
	case errors.As(err, &statusError) && statusError.StatusCode == http.StatusNotFound:
		if err := PullHandler(cmd, args); err != nil {
			return err
		}
	case err != nil:
		return err
	}

	interactive := true

	opts := runOptions{
		Model:    name,
		WordWrap: os.Getenv("TERM") == "xterm-256color",
		Options:  map[string]interface{}{},
	}

	format, err := cmd.Flags().GetString("format")
	if err != nil {
		return err
	}
	opts.Format = format

	prompts := args[1:]

	// prepend stdin to the prompt if provided
	if !term.IsTerminal(int(os.Stdin.Fd())) {
		in, err := io.ReadAll(os.Stdin)
		if err != nil {
			return err
		}

		prompts = append([]string{string(in)}, prompts...)
		opts.WordWrap = false
		interactive = false
	}
	msg := api.Message{
		Role:    "user",
		Content: strings.Join(prompts, " "),
	}
	opts.Messages = append(opts.Messages, msg)
	if len(prompts) > 0 {
		interactive = false
	}

	nowrap, err := cmd.Flags().GetBool("nowordwrap")
	if err != nil {
		return err
	}
	opts.WordWrap = !nowrap

	if !interactive {
		_, err := run(cmd, opts)
		return err
	}

	return runInteractive(cmd, opts)
}

func PushHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	insecure, err := cmd.Flags().GetBool("insecure")
	if err != nil {
		return err
	}

	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	bars := make(map[string]*progress.Bar)
	var status string
	var spinner *progress.Spinner

	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			if spinner != nil {
				spinner.Stop()
			}

			bar, ok := bars[resp.Digest]
			if !ok {
				bar = progress.NewBar(fmt.Sprintf("pushing %s...", resp.Digest[7:19]), resp.Total, resp.Completed)
				bars[resp.Digest] = bar
				p.Add(resp.Digest, bar)
			}

			bar.Set(resp.Completed)
		} else if status != resp.Status {
			if spinner != nil {
				spinner.Stop()
			}

			status = resp.Status
			spinner = progress.NewSpinner(status)
			p.Add(status, spinner)
		}

		return nil
	}

	request := api.PushRequest{Name: args[0], Insecure: insecure}
	if err := client.Push(context.Background(), &request, fn); err != nil {
		return err
	}

	spinner.Stop()
	return nil
}

func ListHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	models, err := client.List(context.Background())
	if err != nil {
		return err
	}

	var data [][]string

	for _, m := range models.Models {
		if len(args) == 0 || strings.HasPrefix(m.Name, args[0]) {
			data = append(data, []string{m.Name, m.Digest[:12], format.HumanBytes(m.Size), format.HumanTime(m.ModifiedAt, "Never")})
		}
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"NAME", "ID", "SIZE", "MODIFIED"})
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.SetHeaderLine(false)
	table.SetBorder(false)
	table.SetNoWhiteSpace(true)
	table.SetTablePadding("\t")
	table.AppendBulk(data)
	table.Render()

	return nil
}

func DeleteHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	for _, name := range args {
		req := api.DeleteRequest{Name: name}
		if err := client.Delete(context.Background(), &req); err != nil {
			return err
		}
		fmt.Printf("deleted '%s'\n", name)
	}
	return nil
}

func ShowHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	if len(args) != 1 {
		return errors.New("missing model name")
	}

	license, errLicense := cmd.Flags().GetBool("license")
	modelfile, errModelfile := cmd.Flags().GetBool("modelfile")
	parameters, errParams := cmd.Flags().GetBool("parameters")
	system, errSystem := cmd.Flags().GetBool("system")
	template, errTemplate := cmd.Flags().GetBool("template")

	for _, boolErr := range []error{errLicense, errModelfile, errParams, errSystem, errTemplate} {
		if boolErr != nil {
			return errors.New("error retrieving flags")
		}
	}

	flagsSet := 0
	showType := ""

	if license {
		flagsSet++
		showType = "license"
	}

	if modelfile {
		flagsSet++
		showType = "modelfile"
	}

	if parameters {
		flagsSet++
		showType = "parameters"
	}

	if system {
		flagsSet++
		showType = "system"
	}

	if template {
		flagsSet++
		showType = "template"
	}

	if flagsSet > 1 {
		return errors.New("only one of '--license', '--modelfile', '--parameters', '--system', or '--template' can be specified")
	} else if flagsSet == 0 {
		return errors.New("one of '--license', '--modelfile', '--parameters', '--system', or '--template' must be specified")
	}

	req := api.ShowRequest{Name: args[0]}
	resp, err := client.Show(context.Background(), &req)
	if err != nil {
		return err
	}

	switch showType {
	case "license":
		fmt.Println(resp.License)
	case "modelfile":
		fmt.Println(resp.Modelfile)
	case "parameters":
		fmt.Println(resp.Parameters)
	case "system":
		fmt.Println(resp.System)
	case "template":
		fmt.Println(resp.Template)
	}

	return nil
}

func CopyHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	req := api.CopyRequest{Source: args[0], Destination: args[1]}
	if err := client.Copy(context.Background(), &req); err != nil {
		return err
	}
	fmt.Printf("copied '%s' to '%s'\n", args[0], args[1])
	return nil
}

func PullHandler(cmd *cobra.Command, args []string) error {
	insecure, err := cmd.Flags().GetBool("insecure")
	if err != nil {
		return err
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	bars := make(map[string]*progress.Bar)

	var status string
	var spinner *progress.Spinner

	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			if spinner != nil {
				spinner.Stop()
			}

			bar, ok := bars[resp.Digest]
			if !ok {
				bar = progress.NewBar(fmt.Sprintf("pulling %s...", resp.Digest[7:19]), resp.Total, resp.Completed)
				bars[resp.Digest] = bar
				p.Add(resp.Digest, bar)
			}

			bar.Set(resp.Completed)
		} else if status != resp.Status {
			if spinner != nil {
				spinner.Stop()
			}

			status = resp.Status
			spinner = progress.NewSpinner(status)
			p.Add(status, spinner)
		}

		return nil
	}

	request := api.PullRequest{Name: args[0], Insecure: insecure}
	if err := client.Pull(context.Background(), &request, fn); err != nil {
		return err
	}

	return nil
}

type runOptions struct {
	Model    string
	Messages []api.Message
	WordWrap bool
	Format   string
	Template string
	Options  map[string]interface{}
}

func run(cmd *cobra.Command, opts runOptions) (*api.Message, error) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, err
	}

	p := progress.NewProgress(os.Stderr)
	defer p.StopAndClear()
	spinner := progress.NewSpinner("")
	p.Add("", spinner)

	termWidth, _, err := term.GetSize(int(os.Stdout.Fd()))
	if err != nil {
		opts.WordWrap = false
	}

	cancelCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT)

	go func() {
		<-sigChan
		cancel()
	}()

	var currentLineLength int
	var wordBuffer string
	var latest api.ChatResponse
	var fullResponse strings.Builder
	var role string

	fn := func(response api.ChatResponse) error {
		p.StopAndClear()
		latest = response
		if response.Message == nil {
			// warm-up response or done
			return nil
		}
		role = response.Message.Role
		content := response.Message.Content
		fullResponse.WriteString(content)

		if opts.WordWrap {
			for _, ch := range content {
				if currentLineLength+1 > termWidth-5 {
					// backtrack the length of the last word and clear to the end of the line
					fmt.Printf("\x1b[%dD\x1b[K\n", len(wordBuffer))
					fmt.Printf("%s%c", wordBuffer, ch)
					currentLineLength = len(wordBuffer) + 1
				} else {
					fmt.Print(string(ch))
					currentLineLength += 1

					switch ch {
					case ' ':
						wordBuffer = ""
					case '\n':
						currentLineLength = 0
					default:
						wordBuffer += string(ch)
					}
				}
			}
		} else {
			fmt.Print(content)
		}

		return nil
	}

	req := &api.ChatRequest{
		Model:    opts.Model,
		Messages: opts.Messages,
		Format:   opts.Format,
		Template: opts.Template,
		Options:  opts.Options,
	}
	if err := client.Chat(cancelCtx, req, fn); err != nil {
		if strings.Contains(err.Error(), "context canceled") {
			return nil, nil
		}
		return nil, err
	}

	if len(opts.Messages) > 0 {
		fmt.Println()
		fmt.Println()
	}

	if !latest.Done {
		return nil, nil
	}

	verbose, err := cmd.Flags().GetBool("verbose")
	if err != nil {
		return nil, err
	}

	if verbose {
		latest.Summary()
	}

	return &api.Message{Role: role, Content: fullResponse.String()}, nil
}

type MultilineState int

const (
	MultilineNone MultilineState = iota
	MultilinePrompt
	MultilineSystem
	MultilineTemplate
)

func runInteractive(cmd *cobra.Command, opts runOptions) error {
	// load the model
	loadOpts := runOptions{Model: opts.Model}
	if _, err := run(cmd, loadOpts); err != nil {
		return err
	}

	usage := func() {
		fmt.Fprintln(os.Stderr, "Available Commands:")
		fmt.Fprintln(os.Stderr, "  /set         Set session variables")
		fmt.Fprintln(os.Stderr, "  /show        Show model information")
		fmt.Fprintln(os.Stderr, "  /bye         Exit")
		fmt.Fprintln(os.Stderr, "  /?, /help    Help for a command")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "Use \"\"\" to begin a multi-line message.")
		fmt.Fprintln(os.Stderr, "")
	}

	usageSet := func() {
		fmt.Fprintln(os.Stderr, "Available Commands:")
		fmt.Fprintln(os.Stderr, "  /set parameter ...     Set a parameter")
		fmt.Fprintln(os.Stderr, "  /set system <string>   Set system prompt")
		fmt.Fprintln(os.Stderr, "  /set template <string> Set prompt template")
		fmt.Fprintln(os.Stderr, "  /set history           Enable history")
		fmt.Fprintln(os.Stderr, "  /set nohistory         Disable history")
		fmt.Fprintln(os.Stderr, "  /set wordwrap          Enable wordwrap")
		fmt.Fprintln(os.Stderr, "  /set nowordwrap        Disable wordwrap")
		fmt.Fprintln(os.Stderr, "  /set format json       Enable JSON mode")
		fmt.Fprintln(os.Stderr, "  /set noformat          Disable formatting")
		fmt.Fprintln(os.Stderr, "  /set verbose           Show LLM stats")
		fmt.Fprintln(os.Stderr, "  /set quiet             Disable LLM stats")
		fmt.Fprintln(os.Stderr, "")
	}

	usageShow := func() {
		fmt.Fprintln(os.Stderr, "Available Commands:")
		fmt.Fprintln(os.Stderr, "  /show license      Show model license")
		fmt.Fprintln(os.Stderr, "  /show modelfile    Show Modelfile for this model")
		fmt.Fprintln(os.Stderr, "  /show parameters   Show parameters for this model")
		fmt.Fprintln(os.Stderr, "  /show system       Show system prompt")
		fmt.Fprintln(os.Stderr, "  /show template     Show prompt template")
		fmt.Fprintln(os.Stderr, "")
	}

	// only list out the most common parameters
	usageParameters := func() {
		fmt.Fprintln(os.Stderr, "Available Parameters:")
		fmt.Fprintln(os.Stderr, "  /set parameter seed <int>             Random number seed")
		fmt.Fprintln(os.Stderr, "  /set parameter num_predict <int>      Max number of tokens to predict")
		fmt.Fprintln(os.Stderr, "  /set parameter top_k <int>            Pick from top k num of tokens")
		fmt.Fprintln(os.Stderr, "  /set parameter top_p <float>          Pick token based on sum of probabilities")
		fmt.Fprintln(os.Stderr, "  /set parameter num_ctx <int>          Set the context size")
		fmt.Fprintln(os.Stderr, "  /set parameter temperature <float>    Set creativity level")
		fmt.Fprintln(os.Stderr, "  /set parameter repeat_penalty <float> How strongly to penalize repetitions")
		fmt.Fprintln(os.Stderr, "  /set parameter repeat_last_n <int>    Set how far back to look for repetitions")
		fmt.Fprintln(os.Stderr, "  /set parameter num_gpu <int>          The number of layers to send to the GPU")
		fmt.Fprintln(os.Stderr, "  /set parameter stop \"<string>\", ...   Set the stop parameters")
		fmt.Fprintln(os.Stderr, "")
	}

	scanner, err := readline.New(readline.Prompt{
		Prompt:         ">>> ",
		AltPrompt:      "... ",
		Placeholder:    "Send a message (/? for help)",
		AltPlaceholder: `Use """ to end multi-line input`,
	})
	if err != nil {
		return err
	}

	fmt.Print(readline.StartBracketedPaste)
	defer fmt.Printf(readline.EndBracketedPaste)

	var multiline MultilineState
	var content string
	var systemContent string
	opts.Messages = make([]api.Message, 0)

	for {
		line, err := scanner.Readline()
		switch {
		case errors.Is(err, io.EOF):
			fmt.Println()
			return nil
		case errors.Is(err, readline.ErrInterrupt):
			if line == "" {
				fmt.Println("\nUse Ctrl-D or /bye to exit.")
			}

			scanner.Prompt.UseAlt = false
			content = ""

			continue
		case err != nil:
			return err
		}

		switch {
		case strings.HasPrefix(content, `"""`):
			// if the prompt so far starts with """ then we're in multiline mode
			// and we need to keep reading until we find a line that ends with """
			cut, found := strings.CutSuffix(line, `"""`)
			content += cut + "\n"

			if !found {
				continue
			}

			content = strings.TrimPrefix(content, `"""`)
			scanner.Prompt.UseAlt = false

			switch multiline {
			case MultilineSystem:
				systemContent = content
				content = ""
				fmt.Println("Set system template.\n")
			case MultilineTemplate:
				opts.Template = content
				content = ""
				fmt.Println("Set model template.\n")
			}
			multiline = MultilineNone
		case strings.HasPrefix(line, `"""`) && len(content) == 0:
			scanner.Prompt.UseAlt = true
			multiline = MultilinePrompt
			content += line + "\n"
			continue
		case scanner.Pasting:
			content += line + "\n"
			continue
		case strings.HasPrefix(line, "/list"):
			args := strings.Fields(line)
			if err := ListHandler(cmd, args[1:]); err != nil {
				return err
			}
		case strings.HasPrefix(line, "/set"):
			args := strings.Fields(line)
			if len(args) > 1 {
				switch args[1] {
				case "history":
					scanner.HistoryEnable()
				case "nohistory":
					scanner.HistoryDisable()
				case "wordwrap":
					opts.WordWrap = true
					fmt.Println("Set 'wordwrap' mode.")
				case "nowordwrap":
					opts.WordWrap = false
					fmt.Println("Set 'nowordwrap' mode.")
				case "verbose":
					cmd.Flags().Set("verbose", "true")
					fmt.Println("Set 'verbose' mode.")
				case "quiet":
					cmd.Flags().Set("verbose", "false")
					fmt.Println("Set 'quiet' mode.")
				case "format":
					if len(args) < 3 || args[2] != "json" {
						fmt.Println("Invalid or missing format. For 'json' mode use '/set format json'")
					} else {
						opts.Format = args[2]
						fmt.Printf("Set format to '%s' mode.\n", args[2])
					}
				case "noformat":
					opts.Format = ""
					fmt.Println("Disabled format.")
				case "parameter":
					if len(args) < 4 {
						usageParameters()
						continue
					}
					var params []string
					for _, p := range args[3:] {
						params = append(params, p)
					}
					fp, err := api.FormatParams(map[string][]string{args[2]: params})
					if err != nil {
						fmt.Printf("Couldn't set parameter: %q\n\n", err)
						continue
					}
					fmt.Printf("Set parameter '%s' to '%s'\n\n", args[2], strings.Join(params, ", "))
					opts.Options[args[2]] = fp[args[2]]
				case "system", "template":
					if len(args) < 3 {
						usageSet()
						continue
					}
					line := strings.Join(args[2:], " ")
					line = strings.TrimPrefix(line, `"""`)
					if strings.HasPrefix(args[2], `"""`) {
						cut, found := strings.CutSuffix(line, `"""`)
						content += cut + "\n"
						if found {
							systemContent = content
							if args[1] == "system" {
								fmt.Println("Set system template.\n")
							} else {
								fmt.Println("Set prompt template.\n")
							}
							content = ""
						} else {
							content = `"""` + content
							if args[1] == "system" {
								multiline = MultilineSystem
							} else {
								multiline = MultilineTemplate
							}
							scanner.Prompt.UseAlt = true
						}
					} else {
						systemContent = line
						fmt.Println("Set system template.\n")
					}
				default:
					fmt.Printf("Unknown command '/set %s'. Type /? for help\n", args[1])
				}
			} else {
				usageSet()
			}
		case strings.HasPrefix(line, "/show"):
			args := strings.Fields(line)
			if len(args) > 1 {
				client, err := api.ClientFromEnvironment()
				if err != nil {
					fmt.Println("error: couldn't connect to ollama server")
					return err
				}
				resp, err := client.Show(cmd.Context(), &api.ShowRequest{Name: opts.Model})
				if err != nil {
					fmt.Println("error: couldn't get model")
					return err
				}

				switch args[1] {
				case "license":
					if resp.License == "" {
						fmt.Print("No license was specified for this model.\n\n")
					} else {
						fmt.Println(resp.License)
					}
				case "modelfile":
					fmt.Println(resp.Modelfile)
				case "parameters":
					if resp.Parameters == "" {
						fmt.Print("No parameters were specified for this model.\n\n")
					} else {
						if len(opts.Options) > 0 {
							fmt.Println("User defined parameters:")
							for k, v := range opts.Options {
								fmt.Printf("%-*s %v\n", 30, k, v)
							}
							fmt.Println()
						}
						fmt.Println("Model defined parameters:")
						fmt.Println(resp.Parameters)
					}
				case "system":
					switch {
					case systemContent != "":
						fmt.Println(systemContent + "\n")
					case resp.System != "":
						fmt.Println(resp.System + "\n")
					default:
						fmt.Print("No system prompt was specified for this model.\n\n")
					}
				case "template":
					switch {
					case opts.Template != "":
						fmt.Println(opts.Template + "\n")
					case resp.Template != "":
						fmt.Println(resp.Template)
					default:
						fmt.Print("No prompt template was specified for this model.\n\n")
					}
				default:
					fmt.Printf("Unknown command '/show %s'. Type /? for help\n", args[1])
				}
			} else {
				usageShow()
			}
		case strings.HasPrefix(line, "/help"), strings.HasPrefix(line, "/?"):
			args := strings.Fields(line)
			if len(args) > 1 {
				switch args[1] {
				case "set", "/set":
					usageSet()
				case "show", "/show":
					usageShow()
				}
			} else {
				usage()
			}
		case line == "/exit", line == "/bye":
			return nil
		case strings.HasPrefix(line, "/"):
			args := strings.Fields(line)
			fmt.Printf("Unknown command '%s'. Type /? for help\n", args[0])
			continue
		default:
			content += line
		}

		if len(content) > 0 && multiline == MultilineNone {
			if systemContent != "" {
				opts.Messages = append(opts.Messages, api.Message{Role: "system", Content: systemContent})
			}
			opts.Messages = append(opts.Messages, api.Message{Role: "user", Content: content})
			assistant, err := run(cmd, opts)
			if err != nil {
				return err
			}
			opts.Messages = append(opts.Messages, *assistant)
		}
	}
}

func RunServer(cmd *cobra.Command, _ []string) error {
	host, port, err := net.SplitHostPort(os.Getenv("OLLAMA_HOST"))
	if err != nil {
		host, port = "127.0.0.1", "11434"
		if ip := net.ParseIP(strings.Trim(os.Getenv("OLLAMA_HOST"), "[]")); ip != nil {
			host = ip.String()
		}
	}

	if err := initializeKeypair(); err != nil {
		return err
	}

	ln, err := net.Listen("tcp", net.JoinHostPort(host, port))
	if err != nil {
		return err
	}

	var origins []string
	if o := os.Getenv("OLLAMA_ORIGINS"); o != "" {
		origins = strings.Split(o, ",")
	}

	return server.Serve(ln, origins)
}

func initializeKeypair() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	privKeyPath := filepath.Join(home, ".ollama", "id_ed25519")
	pubKeyPath := filepath.Join(home, ".ollama", "id_ed25519.pub")

	_, err = os.Stat(privKeyPath)
	if os.IsNotExist(err) {
		fmt.Printf("Couldn't find '%s'. Generating new private key.\n", privKeyPath)
		_, privKey, err := ed25519.GenerateKey(rand.Reader)
		if err != nil {
			return err
		}

		privKeyBytes, err := format.OpenSSHPrivateKey(privKey, "")
		if err != nil {
			return err
		}

		err = os.MkdirAll(filepath.Dir(privKeyPath), 0o755)
		if err != nil {
			return fmt.Errorf("could not create directory %w", err)
		}

		err = os.WriteFile(privKeyPath, pem.EncodeToMemory(privKeyBytes), 0o600)
		if err != nil {
			return err
		}

		sshPrivateKey, err := ssh.NewSignerFromKey(privKey)
		if err != nil {
			return err
		}

		pubKeyData := ssh.MarshalAuthorizedKey(sshPrivateKey.PublicKey())

		err = os.WriteFile(pubKeyPath, pubKeyData, 0o644)
		if err != nil {
			return err
		}

		fmt.Printf("Your new public key is: \n\n%s\n", string(pubKeyData))
	}
	return nil
}

func startMacApp(client *api.Client) error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	link, err := os.Readlink(exe)
	if err != nil {
		return err
	}
	if !strings.Contains(link, "Ollama.app") {
		return fmt.Errorf("could not find ollama app")
	}
	path := strings.Split(link, "Ollama.app")
	if err := exec.Command("/usr/bin/open", "-a", path[0]+"Ollama.app").Run(); err != nil {
		return err
	}
	// wait for the server to start
	timeout := time.After(5 * time.Second)
	tick := time.Tick(500 * time.Millisecond)
	for {
		select {
		case <-timeout:
			return errors.New("timed out waiting for server to start")
		case <-tick:
			if err := client.Heartbeat(context.Background()); err == nil {
				return nil // server has started
			}
		}
	}
}

func checkServerHeartbeat(_ *cobra.Command, _ []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}
	if err := client.Heartbeat(context.Background()); err != nil {
		if !strings.Contains(err.Error(), "connection refused") {
			return err
		}
		if runtime.GOOS == "darwin" {
			if err := startMacApp(client); err != nil {
				return fmt.Errorf("could not connect to ollama app, is it running?")
			}
		} else {
			return fmt.Errorf("could not connect to ollama server, run 'ollama serve' to start it")
		}
	}
	return nil
}

func NewCLI() *cobra.Command {
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	rootCmd := &cobra.Command{
		Use:           "ollama",
		Short:         "Large language model runner",
		SilenceUsage:  true,
		SilenceErrors: true,
		CompletionOptions: cobra.CompletionOptions{
			DisableDefaultCmd: true,
		},
		Version: version.Version,
	}

	cobra.EnableCommandSorting = false

	createCmd := &cobra.Command{
		Use:     "create MODEL",
		Short:   "Create a model from a Modelfile",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    CreateHandler,
	}

	createCmd.Flags().StringP("file", "f", "Modelfile", "Name of the Modelfile (default \"Modelfile\")")

	showCmd := &cobra.Command{
		Use:     "show MODEL",
		Short:   "Show information for a model",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    ShowHandler,
	}

	showCmd.Flags().Bool("license", false, "Show license of a model")
	showCmd.Flags().Bool("modelfile", false, "Show Modelfile of a model")
	showCmd.Flags().Bool("parameters", false, "Show parameters of a model")
	showCmd.Flags().Bool("template", false, "Show template of a model")
	showCmd.Flags().Bool("system", false, "Show system prompt of a model")

	runCmd := &cobra.Command{
		Use:     "run MODEL [PROMPT]",
		Short:   "Run a model",
		Args:    cobra.MinimumNArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    RunHandler,
	}

	runCmd.Flags().Bool("verbose", false, "Show timings for response")
	runCmd.Flags().Bool("insecure", false, "Use an insecure registry")
	runCmd.Flags().Bool("nowordwrap", false, "Don't wrap words to the next line automatically")
	runCmd.Flags().String("format", "", "Response format (e.g. json)")

	serveCmd := &cobra.Command{
		Use:     "serve",
		Aliases: []string{"start"},
		Short:   "Start ollama",
		Args:    cobra.ExactArgs(0),
		RunE:    RunServer,
	}

	pullCmd := &cobra.Command{
		Use:     "pull MODEL",
		Short:   "Pull a model from a registry",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    PullHandler,
	}

	pullCmd.Flags().Bool("insecure", false, "Use an insecure registry")

	pushCmd := &cobra.Command{
		Use:     "push MODEL",
		Short:   "Push a model to a registry",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    PushHandler,
	}

	pushCmd.Flags().Bool("insecure", false, "Use an insecure registry")

	listCmd := &cobra.Command{
		Use:     "list",
		Aliases: []string{"ls"},
		Short:   "List models",
		PreRunE: checkServerHeartbeat,
		RunE:    ListHandler,
	}

	copyCmd := &cobra.Command{
		Use:     "cp SOURCE TARGET",
		Short:   "Copy a model",
		Args:    cobra.ExactArgs(2),
		PreRunE: checkServerHeartbeat,
		RunE:    CopyHandler,
	}

	deleteCmd := &cobra.Command{
		Use:     "rm MODEL [MODEL...]",
		Short:   "Remove a model",
		Args:    cobra.MinimumNArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    DeleteHandler,
	}

	rootCmd.AddCommand(
		serveCmd,
		createCmd,
		showCmd,
		runCmd,
		pullCmd,
		pushCmd,
		listCmd,
		copyCmd,
		deleteCmd,
	)

	return rootCmd
}
